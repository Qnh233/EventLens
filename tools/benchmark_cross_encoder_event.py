from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from benchmark_routed_subject_svc_v2 import routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.cross_encoder_event import (
    build_article_text,
    build_training_pairs,
    choose_ranked_events,
    merge_candidate_names,
)
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel


def descriptions_for_scope(schema: EventSchemaIndex, scope: str) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for row in schema.definitions:
        if row.scope != scope:
            continue
        values = grouped.setdefault(row.event_name, [])
        value = str(row.description or "").strip()
        if value and value not in values:
            values.append(value)
    return {label: "；".join(values[:2])[:900] for label, values in grouped.items()}


def metrics(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def build_candidate_sets(scores, classes, recalls, *, svc_k=5, bge_k=5):
    order = np.argsort(-scores, axis=1)
    output = []
    for index, recall in enumerate(recalls):
        svc = [classes[j] for j in order[index, :svc_k]]
        bge = [row.event_name for row in recall.candidates[:bge_k]]
        output.append(merge_candidate_names(svc, bge, limit=10))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company"], default="company")
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--model", default="hfl/chinese-macbert-base")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or test_manifest.article_count != len(test):
        raise ValueError("embedding 数量不一致")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding 顺序不一致")
    if load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
        raise ValueError("test embedding 顺序不一致")

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    native = settings.native_embedding
    embedding_client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    train_routes, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope="company", settings=settings, schema=schema, client=embedding_client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope="company", settings=settings, schema=schema, client=embedding_client
    )

    truths = [str(row.event_label) for row in train]
    test_truth = [str(row.event_label) for row in test]
    classes = sorted(set(truths))
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    for fit_idx, val_idx in folds.split(train_texts, truths):
        svc = fit_with_descriptions(
            [train_texts[i] for i in fit_idx],
            [truths[i] for i in fit_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        oof_scores[val_idx] = decision_scores(svc, [train_texts[i] for i in val_idx], classes)
    full_svc = fit_with_descriptions(train_texts, truths, desc_texts, desc_labels, repeat=1)
    test_scores = decision_scores(full_svc, test_texts, classes)
    train_candidates = build_candidate_sets(oof_scores, classes, train_recalls)
    test_candidates = build_candidate_sets(test_scores, classes, test_recalls)
    train_candidate_hit = sum(label in candidates for label, candidates in zip(truths, train_candidates)) / len(train)
    test_candidate_hit = sum(label in candidates for label, candidates in zip(test_truth, test_candidates)) / len(test)

    descriptions = descriptions_for_scope(schema, "company")
    train_idx, val_idx = train_test_split(
        list(range(len(train))),
        test_size=0.2,
        random_state=settings.model.random_state,
        stratify=truths,
    )
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PairDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, index):
            return self.pairs[index]

    def collate(pairs):
        encoded = tokenizer(
            [pair.article_text for pair in pairs],
            [pair.event_text for pair in pairs],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["targets"] = torch.tensor([pair.target for pair in pairs], dtype=torch.float32)
        return encoded

    def candidate_scores(model, articles, indices, candidates):
        pairs = []
        group_sizes = []
        for index in indices:
            names = candidates[index]
            group_sizes.append(len(names))
            article_text = build_article_text(articles[index], max_content_chars=1800)
            for name in names:
                pairs.append((article_text, f"事件类型：{name}\n事件定义：{descriptions.get(name, '')}"))
        scores = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(pairs), args.batch_size):
                batch = pairs[start : start + args.batch_size]
                encoded = tokenizer(
                    [row[0] for row in batch],
                    [row[1] for row in batch],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                logits = model(**encoded).logits.squeeze(-1)
                scores.extend(logits.detach().cpu().tolist())
        grouped = []
        offset = 0
        for size in group_sizes:
            grouped.append(scores[offset : offset + size])
            offset += size
        return grouped

    def train_model(indices, epochs):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            local_files_only=True,
        ).to(device)
        pairs = build_training_pairs(
            train,
            truths,
            train_candidates,
            descriptions,
            indices=indices,
            max_negatives=5,
            max_content_chars=1800,
        )
        positives = sum(pair.target > 0.5 for pair in pairs)
        negatives = len(pairs) - positives
        pos_weight = torch.tensor([negatives / max(1, positives)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        loader = DataLoader(
            PairDataset(pairs),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        for _ in range(epochs):
            model.train()
            for batch in loader:
                targets = batch.pop("targets").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs).logits.squeeze(-1)
                loss = loss_fn(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        return model, len(pairs)

    history = []
    best_epoch = 1
    best_f1 = -1.0
    best_state = None
    started = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        local_files_only=True,
    ).to(device)
    pairs = build_training_pairs(
        train,
        truths,
        train_candidates,
        descriptions,
        indices=train_idx,
        max_negatives=5,
        max_content_chars=1800,
    )
    positives = sum(pair.target > 0.5 for pair in pairs)
    negatives = len(pairs) - positives
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(1, positives)], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader = DataLoader(PairDataset(pairs), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            targets = batch.pop("targets").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits.squeeze(-1)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        val_scores = candidate_scores(model, train, val_idx, train_candidates)
        val_names = [train_candidates[index] for index in val_idx]
        val_pred = choose_ranked_events(val_names, val_scores)
        val_truth = [truths[index] for index in val_idx]
        row = {"epoch": epoch, "train_loss": round(total_loss / max(1, len(loader)), 6), **metrics(val_truth, val_pred)}
        history.append(row)
        if row["macro_f1"] > best_f1:
            best_f1 = row["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    validation_seconds = time.perf_counter() - started

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_model, final_pair_count = train_model(list(range(len(train))), best_epoch)
    test_rank_scores = candidate_scores(final_model, test, list(range(len(test))), test_candidates)
    test_pred = choose_ranked_events(test_candidates, test_rank_scores)
    external = metrics(test_truth, test_pred)
    payload = {
        "scope": "company",
        "protocol": "production-like label-description cross-encoder; OOF SVC Top5 + BGE Top5 candidates; article-level validation split",
        "model": args.model,
        "train_count": len(train),
        "external_count": len(test),
        "train_candidate_hit_rate": round(train_candidate_hit, 6),
        "external_candidate_hit_rate": round(test_candidate_hit, 6),
        "validation_history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": round(best_f1, 6),
        "final_pair_count": final_pair_count,
        "external": external,
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
        "selection_training_seconds": round(validation_seconds, 2),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
