from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_pretrained_reranker_cascade import constrained
from benchmark_routed_subject_svc_v2 import metric
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.score_calibration import ClasswiseScoreCalibrator


METHODS = ("identity", "zscore", "platt")


def _article_text(row) -> str:
    return f"{row.title}。{row.content}"


def _label_indices(labels: list[str], classes: list[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    label_indices = _label_indices(labels, classes)
    train_texts = [_article_text(row) for row in train]
    test_texts = [_article_text(row) for row in test]

    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
        raise ValueError("test embedding order mismatch")

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    native = settings.native_embedding
    client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    train_routes, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope="company", settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope="company", settings=settings, schema=schema, client=client
    )

    base_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    for fit_idx, val_idx in base_splitter.split(train_texts, labels):
        model = fit_with_descriptions(
            [train_texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        oof_scores[val_idx] = decision_scores(model, [train_texts[i] for i in val_idx], classes)

    meta_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state + 17)
    meta_scores = {method: np.zeros_like(oof_scores) for method in METHODS}
    for fit_idx, val_idx in meta_splitter.split(oof_scores, labels):
        for method in METHODS:
            calibrator = ClasswiseScoreCalibrator(method).fit(oof_scores[fit_idx], label_indices[fit_idx])
            meta_scores[method][val_idx] = calibrator.transform(oof_scores[val_idx])

    oof = {}
    for method in METHODS:
        pred = constrained(classes, meta_scores[method], train_routes, train_recalls, schema)
        oof[method] = metric(labels, pred)
    selected_method = max(METHODS, key=lambda method: oof[method]["macro_f1"])

    report = {
        "scope": "company",
        "protocol": "production-like nested-OOF classwise calibration of char-SVC OVR scores; calibration family selected only by meta-OOF; no labeled-only subject truth; external untouched unless non-identity improves train meta-OOF",
        "oof": oof,
        "selected_method": selected_method,
        "external_evaluated": False,
        "production_like_svc_reference_macro_f1": 0.770798,
        "gate_macro_f1": 0.80,
        "stretch_target_macro_f1": 0.85,
    }

    identity_score = oof["identity"]["macro_f1"]
    selected_score = oof[selected_method]["macro_f1"]
    if selected_method != "identity" and selected_score > identity_score:
        full_model = fit_with_descriptions(train_texts, labels, desc_texts, desc_labels, repeat=1)
        external_scores = decision_scores(full_model, test_texts, classes)
        calibrator = ClasswiseScoreCalibrator(selected_method).fit(oof_scores, label_indices)
        calibrated_external = calibrator.transform(external_scores)
        external_pred = constrained(classes, calibrated_external, test_routes, test_recalls, schema)
        external = metric(test_labels, external_pred)
        report.update(
            external_evaluated=True,
            external=external,
            gain_vs_reference=round(external["macro_f1"] - 0.770798, 6),
            gate_passed=bool(external["macro_f1"] >= 0.80),
        )

    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
