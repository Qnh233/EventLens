from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from benchmark_cross_encoder_event import descriptions_for_scope
from benchmark_pretrained_reranker_cascade import (
    build_candidate_sets,
    candidate_rank_matrix,
    constrained,
    gated_matrix,
    score_candidates,
)
from benchmark_routed_subject_svc_v2 import metric, routed_text
from benchmark_schema_constrained_svc import (
    build_model,
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.groupwise_event_reranker import (
    build_training_group,
    class_sample_weights,
    freeze_reranker_except_last_layers,
)
from eventlens.hard_negative_mining import build_confusion_hard_negative_map
from eventlens.io import read_competition_labeled_excel
from eventlens.score_fusion import top1_margin


TRAINABLE_LAST_LAYERS = (2, 4)
LEARNING_RATES = (1e-5,)
MAX_EPOCHS = 3
TRAIN_NEGATIVES = 3
BLEND_WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MARGIN_QUANTILES = (0.1, 0.2, 0.3)


def top_names(scores: np.ndarray, classes: list[str], *, k: int) -> list[list[str]]:
    order = np.argsort(-np.asarray(scores), axis=1)
    return [[classes[j] for j in row[:k]] for row in order]


def bge_names(recalls, *, k: int) -> list[list[str]]:
    return [[candidate.event_name for candidate in row.candidates[:k]] for row in recalls]


def make_training_groups(
    labels: list[str],
    svc_candidates: list[list[str]],
    recall_candidates: list[list[str]],
    hard_map: dict[int, list[int]],
    classes: list[str],
    *,
    max_negatives: int,
) -> list[list[str]]:
    class_to_id = {label: index for index, label in enumerate(classes)}
    groups: list[list[str]] = []
    for truth, svc, recall in zip(labels, svc_candidates, recall_candidates):
        class_id = class_to_id[truth]
        hard = [classes[index] for index in hard_map[class_id]]
        groups.append(
            build_training_group(
                truth,
                svc_candidates=svc,
                bge_candidates=recall,
                hard_candidates=hard,
                max_negatives=max_negatives,
            )
        )
    return groups


def oof_svc_scores(
    texts: list[str],
    labels: list[str],
    classes: list[str],
    desc_texts: list[str],
    desc_labels: list[str],
    *,
    random_state: int,
) -> np.ndarray:
    output = np.full((len(texts), len(classes)), -1e9, dtype=np.float64)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    texts_array = np.asarray(texts, dtype=object)
    labels_array = np.asarray(labels, dtype=object)
    for fit_idx, val_idx in folds.split(texts_array, labels_array):
        model = fit_with_descriptions(
            texts_array[fit_idx].tolist(),
            labels_array[fit_idx].tolist(),
            desc_texts,
            desc_labels,
            repeat=1,
        )
        output[val_idx] = decision_scores(model, texts_array[val_idx].tolist(), classes)
    return output


def train_groupwise(
    *,
    model_path: str,
    tokenizer,
    articles,
    labels: list[str],
    candidate_groups: list[list[str]],
    descriptions: dict[str, str],
    last_layers: int,
    learning_rate: float,
    epochs: int,
    class_weight_power: float,
    max_length: int,
    max_content_chars: int,
    batch_articles: int,
    random_state: int,
    device,
    torch_module,
    validation_callback=None,
):
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    torch = torch_module
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    ).to(device)
    trainable, total = freeze_reranker_except_last_layers(model, last_n=last_layers)
    sample_weights = class_sample_weights(labels, power=class_weight_power)

    class GroupDataset(Dataset):
        def __len__(self):
            return len(articles)

        def __getitem__(self, index):
            return index

    def collate(indices):
        left: list[str] = []
        right: list[str] = []
        widths: list[int] = []
        weights: list[float] = []
        for index in indices:
            article_text = routed_text(
                articles[index], None, mode="no_subject", max_content_chars=max_content_chars
            )
            candidates = candidate_groups[index]
            widths.append(len(candidates))
            weights.append(float(sample_weights[index]))
            for label in candidates:
                label_text = f"事件类型：{label}\n事件定义：{descriptions.get(label, '')}"
                # Reranker pretraining treats the shorter semantic intent as query.
                left.append(label_text)
                right.append(article_text)
        if len(set(widths)) != 1:
            raise ValueError("training candidate groups must have equal width")
        encoded = tokenizer(
            left,
            right,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded, len(widths), widths[0], torch.tensor(weights, dtype=torch.float32)

    loader = DataLoader(
        GroupDataset(),
        batch_size=batch_articles,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(random_state),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict] = []
    best_state = None
    best_score = -1.0
    best_epoch = epochs
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        correct = 0
        for encoded, batch_size, width, weights in loader:
            inputs = {key: value.to(device) for key, value in encoded.items()}
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**inputs, return_dict=True).logits.reshape(batch_size, width).float()
                targets = torch.zeros(batch_size, dtype=torch.long, device=device)
                losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            correct += int((logits.argmax(dim=1) == 0).sum().item())
            seen += batch_size

        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, len(loader)), 6),
            "group_accuracy": round(correct / max(1, seen), 6),
        }
        if validation_callback is not None:
            validation = validation_callback(model)
            row.update(validation)
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_macro_f1": best_score,
        "training_seconds": round(training_seconds, 2),
        "peak_vram_mb": round(float(peak_vram), 1),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": round(trainable / max(1, total), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-content-chars", type=int, default=2400)
    parser.add_argument("--batch-articles", type=int, default=2)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=args.max_content_chars)
        for row in train
    ]
    test_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=args.max_content_chars)
        for row in test
    ]

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    descriptions = descriptions_for_scope(schema, "company")

    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
        raise ValueError("test embedding order mismatch")

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
    description_by_label = dict(zip(desc_labels, desc_texts))
    anchor_vectors = np.asarray(client.embed([description_by_label[label] for label in classes]), dtype=np.float32)

    indices = np.arange(len(train))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=settings.model.random_state,
        stratify=labels,
    )
    fit_idx = np.asarray(sorted(fit_idx))
    val_idx = np.asarray(sorted(val_idx))
    fit_texts = [train_texts[i] for i in fit_idx]
    fit_labels = [labels[i] for i in fit_idx]

    fit_oof = oof_svc_scores(
        fit_texts,
        fit_labels,
        classes,
        desc_texts,
        desc_labels,
        random_state=settings.model.random_state,
    )
    fit_svc_candidates = top_names(fit_oof, classes, k=5)
    fit_recall_candidates = bge_names([train_recalls[i] for i in fit_idx], k=5)
    fit_hard_map = build_confusion_hard_negative_map(
        fit_texts,
        fit_labels,
        classes,
        anchor_vectors,
        model_factory=build_model,
        top_k=3,
        semantic_weight=0.5,
        random_state=settings.model.random_state,
    )
    fit_groups = make_training_groups(
        fit_labels,
        fit_svc_candidates,
        fit_recall_candidates,
        fit_hard_map,
        classes,
        max_negatives=TRAIN_NEGATIVES,
    )

    fit_svc = fit_with_descriptions(fit_texts, fit_labels, desc_texts, desc_labels, repeat=1)
    val_svc_scores = decision_scores(fit_svc, [train_texts[i] for i in val_idx], classes)
    val_candidates = build_candidate_sets(
        val_svc_scores,
        classes,
        [train_recalls[i] for i in val_idx],
        svc_k=5,
        bge_k=5,
    )
    val_truth = [labels[i] for i in val_idx]
    val_candidate_hit = sum(
        truth in group for truth, group in zip(val_truth, val_candidates)
    ) / len(val_truth)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search: dict[str, dict] = {}
    best = None

    for last_layers in TRAINABLE_LAST_LAYERS:
        for learning_rate in LEARNING_RATES:
            def validate(model):
                rerank_scores = score_candidates(
                    model,
                    tokenizer,
                    [train[i] for i in val_idx],
                    val_candidates,
                    descriptions,
                    orientation="label_to_article",
                    batch_size=16,
                    max_length=args.max_length,
                    max_content_chars=args.max_content_chars,
                    device=device,
                    torch_module=torch,
                )
                dense = candidate_rank_matrix(
                    val_svc_scores,
                    rerank_scores,
                    val_candidates,
                    classes,
                    reranker_weight=1.0,
                )
                pred = constrained(
                    classes,
                    dense,
                    [train_routes[i] for i in val_idx],
                    [train_recalls[i] for i in val_idx],
                    schema,
                )
                return metric(val_truth, pred)

            result = train_groupwise(
                model_path=args.model,
                tokenizer=tokenizer,
                articles=[train[i] for i in fit_idx],
                labels=fit_labels,
                candidate_groups=fit_groups,
                descriptions=descriptions,
                last_layers=last_layers,
                learning_rate=learning_rate,
                epochs=MAX_EPOCHS,
                class_weight_power=0.5,
                max_length=args.max_length,
                max_content_chars=args.max_content_chars,
                batch_articles=args.batch_articles,
                random_state=settings.model.random_state,
                device=device,
                torch_module=torch,
                validation_callback=validate,
            )
            key = f"last{last_layers}:lr{learning_rate:g}"
            search[key] = {
                key_name: value
                for key_name, value in result.items()
                if key_name != "model"
            }
            candidate = {
                "key": key,
                "last_layers": last_layers,
                "learning_rate": learning_rate,
                "best_epoch": result["best_epoch"],
                "macro_f1": result["best_macro_f1"],
            }
            if best is None or candidate["macro_f1"] > best["macro_f1"]:
                best = candidate
            del result["model"]
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("no groupwise candidate completed")

    # Refit the selected model on all Gold using OOF candidates, keeping test
    # completely outside model and policy selection.
    full_oof = oof_svc_scores(
        train_texts,
        labels,
        classes,
        desc_texts,
        desc_labels,
        random_state=settings.model.random_state,
    )
    full_hard_map = build_confusion_hard_negative_map(
        train_texts,
        labels,
        classes,
        anchor_vectors,
        model_factory=build_model,
        top_k=3,
        semantic_weight=0.5,
        random_state=settings.model.random_state,
    )
    full_groups = make_training_groups(
        labels,
        top_names(full_oof, classes, k=5),
        bge_names(train_recalls, k=5),
        full_hard_map,
        classes,
        max_negatives=TRAIN_NEGATIVES,
    )
    final = train_groupwise(
        model_path=args.model,
        tokenizer=tokenizer,
        articles=train,
        labels=labels,
        candidate_groups=full_groups,
        descriptions=descriptions,
        last_layers=best["last_layers"],
        learning_rate=best["learning_rate"],
        epochs=best["best_epoch"],
        class_weight_power=0.5,
        max_length=args.max_length,
        max_content_chars=args.max_content_chars,
        batch_articles=args.batch_articles,
        random_state=settings.model.random_state,
        device=device,
        torch_module=torch,
    )

    # Select the light-weight fusion/deferral rule on the same train-only
    # validation split, using the already selected reranker configuration.
    selected_model = final["model"]
    # For policy selection, retrain on fit only with the selected epoch count.
    policy_model = train_groupwise(
        model_path=args.model,
        tokenizer=tokenizer,
        articles=[train[i] for i in fit_idx],
        labels=fit_labels,
        candidate_groups=fit_groups,
        descriptions=descriptions,
        last_layers=best["last_layers"],
        learning_rate=best["learning_rate"],
        epochs=best["best_epoch"],
        class_weight_power=0.5,
        max_length=args.max_length,
        max_content_chars=args.max_content_chars,
        batch_articles=args.batch_articles,
        random_state=settings.model.random_state,
        device=device,
        torch_module=torch,
    )["model"]
    val_rerank_scores = score_candidates(
        policy_model,
        tokenizer,
        [train[i] for i in val_idx],
        val_candidates,
        descriptions,
        orientation="label_to_article",
        batch_size=16,
        max_length=args.max_length,
        max_content_chars=args.max_content_chars,
        device=device,
        torch_module=torch,
    )
    policy_search: dict[str, dict] = {}
    best_policy = {"family": "svc", "key": "svc", "macro_f1": metric(val_truth, constrained(classes, val_svc_scores, [train_routes[i] for i in val_idx], [train_recalls[i] for i in val_idx], schema))["macro_f1"]}
    for weight in BLEND_WEIGHTS:
        dense = candidate_rank_matrix(
            val_svc_scores,
            val_rerank_scores,
            val_candidates,
            classes,
            reranker_weight=weight,
        )
        pred = constrained(classes, dense, [train_routes[i] for i in val_idx], [train_recalls[i] for i in val_idx], schema)
        score = metric(val_truth, pred)
        key = f"w={weight}"
        policy_search[f"blend:{key}"] = score
        if score["macro_f1"] > best_policy["macro_f1"]:
            best_policy = {"family": "blend", "key": key, "weight": weight, "macro_f1": score["macro_f1"]}

    reranker_only = candidate_rank_matrix(
        val_svc_scores,
        val_rerank_scores,
        val_candidates,
        classes,
        reranker_weight=1.0,
    )
    margins = top1_margin(val_svc_scores)
    for quantile in MARGIN_QUANTILES:
        threshold = float(np.quantile(margins, quantile))
        dense = gated_matrix(val_svc_scores, reranker_only, margin_threshold=threshold)
        pred = constrained(classes, dense, [train_routes[i] for i in val_idx], [train_recalls[i] for i in val_idx], schema)
        score = metric(val_truth, pred)
        key = f"q={quantile}"
        policy_search[f"gate:{key}"] = {**score, "threshold": round(threshold, 6)}
        if score["macro_f1"] > best_policy["macro_f1"]:
            best_policy = {"family": "gate", "key": key, "threshold": threshold, "macro_f1": score["macro_f1"]}

    full_svc = fit_with_descriptions(train_texts, labels, desc_texts, desc_labels, repeat=1)
    test_svc_scores = decision_scores(full_svc, test_texts, classes)
    test_candidates = build_candidate_sets(test_svc_scores, classes, test_recalls, svc_k=5, bge_k=5)
    test_candidate_hit = sum(
        truth in group for truth, group in zip(test_labels, test_candidates)
    ) / len(test_labels)
    test_rerank_scores = score_candidates(
        selected_model,
        tokenizer,
        test,
        test_candidates,
        descriptions,
        orientation="label_to_article",
        batch_size=16,
        max_length=args.max_length,
        max_content_chars=args.max_content_chars,
        device=device,
        torch_module=torch,
    )
    if best_policy["family"] == "blend":
        final_scores = candidate_rank_matrix(
            test_svc_scores,
            test_rerank_scores,
            test_candidates,
            classes,
            reranker_weight=float(best_policy["weight"]),
        )
    elif best_policy["family"] == "gate":
        reranker_dense = candidate_rank_matrix(
            test_svc_scores,
            test_rerank_scores,
            test_candidates,
            classes,
            reranker_weight=1.0,
        )
        final_scores = gated_matrix(
            test_svc_scores,
            reranker_dense,
            margin_threshold=float(best_policy["threshold"]),
        )
    else:
        final_scores = test_svc_scores
    test_pred = constrained(classes, final_scores, test_routes, test_recalls, schema)
    external = metric(test_labels, test_pred)
    svc_reference = metric(
        test_labels,
        constrained(classes, test_svc_scores, test_routes, test_recalls, schema),
    )

    payload = {
        "scope": "company",
        "protocol": "production-like groupwise BGE-reranker-v2-m3 fine-tuning; 80/20 train-only model+epoch+policy selection; training negatives from fit-only/full 3-fold OOF SVC confusion + BGE recall + Schema semantics; external touched once",
        "model": args.model,
        "fit_count": len(fit_idx),
        "validation_count": len(val_idx),
        "validation_candidate_hit_rate": round(val_candidate_hit, 6),
        "search": search,
        "selected_model": best,
        "policy_search": policy_search,
        "selected_policy": best_policy,
        "final_training": {key: value for key, value in final.items() if key != "model"},
        "external_candidate_hit_rate": round(test_candidate_hit, 6),
        "svc_external": svc_reference,
        "external": external,
        "gain_vs_svc": round(external["macro_f1"] - svc_reference["macro_f1"], 6),
        "gate_macro_f1": 0.80,
        "stretch_target_macro_f1": 0.85,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
