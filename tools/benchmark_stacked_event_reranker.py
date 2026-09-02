from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from benchmark_routed_subject_svc_v2 import metric, predict_with_policy, routed_text
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


@dataclass(frozen=True)
class CandidateConfig:
    svc_top_k: int
    bge_top_k: int

    @property
    def key(self) -> str:
        return f"svc{self.svc_top_k}_bge{self.bge_top_k}"


def _candidate_labels(
    score_row: np.ndarray,
    classes: list[str],
    recall,
    *,
    svc_top_k: int,
    bge_top_k: int,
) -> list[str]:
    order = np.argsort(-score_row)
    output = [classes[index] for index in order[:svc_top_k]]
    for candidate in recall.candidates[:bge_top_k]:
        if candidate.event_name in classes and candidate.event_name not in output:
            output.append(candidate.event_name)
    return output


def _feature_vector(
    label: str,
    *,
    score_row: np.ndarray,
    classes: list[str],
    recall,
    route,
) -> list[float]:
    class_index = classes.index(label)
    svc_order = np.argsort(-score_row)
    svc_rank = int(np.where(svc_order == class_index)[0][0]) + 1
    top_score = float(score_row[svc_order[0]])
    mean_score = float(np.mean(score_row))

    bge_by_label: dict[str, tuple[float, int]] = {}
    for rank, candidate in enumerate(recall.candidates, 1):
        current = bge_by_label.get(candidate.event_name)
        if current is None or candidate.score > current[0]:
            bge_by_label[candidate.event_name] = (float(candidate.score), rank)
    bge_score, bge_rank = bge_by_label.get(label, (-1.0, 0))

    # 所有特征在真实生产推理时均可获得；最后的 class one-hot 只做类级校准。
    numeric = [
        float(score_row[class_index]),
        float(score_row[class_index]) - mean_score,
        float(score_row[class_index]) - top_score,
        1.0 / svc_rank,
        float(svc_rank == 1),
        float(svc_rank <= 3),
        float(bge_score),
        (1.0 / bge_rank) if bge_rank else 0.0,
        float(bge_rank == 1),
        float(0 < bge_rank <= 3),
        float(0 < bge_rank <= 5),
        float(route.method == "exact_alias"),
        float(route.accepted_subject_code is not None),
    ]
    one_hot = [0.0] * len(classes)
    one_hot[class_index] = 1.0
    return numeric + one_hot


def build_pair_dataset(
    scores: np.ndarray,
    classes: list[str],
    routes,
    recalls,
    *,
    truth: list[str] | None,
    config: CandidateConfig,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[list[str]], float | None]:
    rows: list[list[float]] = []
    targets: list[int] = []
    article_indices: list[int] = []
    article_candidates: list[list[str]] = []
    hit_count = 0
    for article_index, (score_row, route, recall) in enumerate(zip(scores, routes, recalls)):
        candidates = _candidate_labels(
            score_row,
            classes,
            recall,
            svc_top_k=config.svc_top_k,
            bge_top_k=config.bge_top_k,
        )
        article_candidates.append(candidates)
        if truth is not None:
            hit_count += int(truth[article_index] in candidates)
        for label in candidates:
            rows.append(
                _feature_vector(
                    label,
                    score_row=score_row,
                    classes=classes,
                    recall=recall,
                    route=route,
                )
            )
            article_indices.append(article_index)
            if truth is not None:
                targets.append(int(label == truth[article_index]))
    matrix = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int8) if truth is not None else None
    hit_rate = (hit_count / len(scores)) if truth is not None else None
    return matrix, y, np.asarray(article_indices, dtype=np.int32), article_candidates, hit_rate


def rank_articles(
    model: LogisticRegression,
    features: np.ndarray,
    article_indices: np.ndarray,
    article_candidates: list[list[str]],
) -> list[str]:
    pair_scores = model.decision_function(features)
    output: list[str] = []
    for article_index, candidates in enumerate(article_candidates):
        row_positions = np.flatnonzero(article_indices == article_index)
        if not len(row_positions):
            raise ValueError(f"文章 {article_index} 没有候选")
        best_local = int(np.argmax(pair_scores[row_positions]))
        output.append(candidates[best_local])
    return output


def _score(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(truth, pred)), 6),
        "macro_f1": round(float(f1_score(truth, pred, average="macro", zero_division=0)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company"], default="company")
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [
        row.article_id for row in train
    ]:
        raise ValueError("train embedding 顺序不一致")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [
        row.article_id for row in test
    ]:
        raise ValueError("test embedding 顺序不一致")

    native = settings.native_embedding
    client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    train_routes, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope="company", settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope="company", settings=settings, schema=schema, client=client
    )

    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    description_texts, description_labels = label_description_examples(schema, scope="company")
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train
    ]
    test_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test
    ]

    base_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    for train_idx, val_idx in base_folds.split(train_texts, labels):
        model = fit_with_descriptions(
            [train_texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            description_texts,
            description_labels,
            repeat=1,
        )
        oof_scores[val_idx] = decision_scores(
            model, [train_texts[i] for i in val_idx], classes
        )

    baseline_oof = predict_with_policy(
        classes,
        oof_scores,
        train_routes,
        train_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )

    configs = [
        CandidateConfig(svc_k, bge_k)
        for svc_k in (3, 5, 8)
        for bge_k in (3, 5)
    ]
    reranker_params = [
        (c_value, class_weight)
        for c_value in (0.1, 0.3, 1.0, 3.0, 10.0)
        for class_weight in (None, "balanced")
    ]
    meta_folds = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=settings.model.random_state + 17
    )
    candidates_report: dict[str, dict] = {}
    winner: tuple[CandidateConfig, float, str | None] | None = None
    winner_macro_f1 = -1.0
    winner_std = float("inf")

    for config in configs:
        features, targets, article_indices, article_candidates, hit_rate = build_pair_dataset(
            oof_scores,
            classes,
            train_routes,
            train_recalls,
            truth=labels,
            config=config,
        )
        assert targets is not None and hit_rate is not None
        config_rows: list[dict] = []
        for c_value, class_weight in reranker_params:
            fold_scores: list[float] = []
            fold_accuracy: list[float] = []
            for train_articles, val_articles in meta_folds.split(np.zeros(len(labels)), labels):
                train_mask = np.isin(article_indices, train_articles)
                val_mask = np.isin(article_indices, val_articles)
                model = LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=settings.model.random_state,
                )
                model.fit(features[train_mask], targets[train_mask])

                val_article_map = {int(old): new for new, old in enumerate(val_articles)}
                val_pair_article_indices = np.asarray(
                    [val_article_map[int(index)] for index in article_indices[val_mask]], dtype=np.int32
                )
                val_candidates = [article_candidates[int(index)] for index in val_articles]
                pred = rank_articles(
                    model,
                    features[val_mask],
                    val_pair_article_indices,
                    val_candidates,
                )
                truth = [labels[int(index)] for index in val_articles]
                metrics = _score(truth, pred)
                fold_scores.append(metrics["macro_f1"])
                fold_accuracy.append(metrics["accuracy"])
            row = {
                "C": c_value,
                "class_weight": class_weight,
                "macro_f1_mean": round(float(np.mean(fold_scores)), 6),
                "macro_f1_std": round(float(np.std(fold_scores)), 6),
                "accuracy_mean": round(float(np.mean(fold_accuracy)), 6),
                "fold_macro_f1": fold_scores,
            }
            config_rows.append(row)
            if (
                row["macro_f1_mean"] > winner_macro_f1
                or (
                    row["macro_f1_mean"] == winner_macro_f1
                    and row["macro_f1_std"] < winner_std
                )
            ):
                winner = (config, c_value, class_weight)
                winner_macro_f1 = row["macro_f1_mean"]
                winner_std = row["macro_f1_std"]
        candidates_report[config.key] = {
            "candidate_hit_rate": round(hit_rate, 6),
            "mean_candidate_count": round(
                float(np.mean([len(row) for row in article_candidates])), 6
            ),
            "rerankers": config_rows,
        }

    assert winner is not None
    winner_config, winner_c, winner_class_weight = winner
    winner_features, winner_targets, winner_article_indices, _, _ = build_pair_dataset(
        oof_scores,
        classes,
        train_routes,
        train_recalls,
        truth=labels,
        config=winner_config,
    )
    assert winner_targets is not None
    reranker = LogisticRegression(
        C=winner_c,
        class_weight=winner_class_weight,
        solver="liblinear",
        max_iter=1000,
        random_state=settings.model.random_state,
    )
    reranker.fit(winner_features, winner_targets)

    base_model = fit_with_descriptions(
        train_texts,
        labels,
        description_texts,
        description_labels,
        repeat=1,
    )
    test_scores = decision_scores(base_model, test_texts, classes)
    baseline_external = predict_with_policy(
        classes,
        test_scores,
        test_routes,
        test_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    test_features, _, test_article_indices, test_candidates, test_hit_rate = build_pair_dataset(
        test_scores,
        classes,
        test_routes,
        test_recalls,
        truth=test_labels,
        config=winner_config,
    )
    stacked_external = rank_articles(
        reranker,
        test_features,
        test_article_indices,
        test_candidates,
    )

    payload = {
        "scope": "company",
        "protocol": "production-like OOF stacking; no true subject fields in classifier/reranker; base and reranker choices use train OOF/meta-CV only",
        "base_oof": metric(labels, baseline_oof),
        "winner": {
            "candidate_config": winner_config.key,
            "svc_top_k": winner_config.svc_top_k,
            "bge_top_k": winner_config.bge_top_k,
            "C": winner_c,
            "class_weight": winner_class_weight,
            "meta_cv_macro_f1_mean": winner_macro_f1,
            "meta_cv_macro_f1_std": winner_std,
        },
        "candidate_search": candidates_report,
        "external_candidate_hit_rate": round(float(test_hit_rate or 0.0), 6),
        "baseline_external": metric(test_labels, baseline_external),
        "stacked_external": metric(test_labels, stacked_external),
        "macro_f1_gain": round(
            metric(test_labels, stacked_external)["macro_f1"]
            - metric(test_labels, baseline_external)["macro_f1"],
            6,
        ),
        "gate_macro_f1": 0.80,
        "gate_passed": metric(test_labels, stacked_external)["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
