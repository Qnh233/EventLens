from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventlens.config import load_settings
from eventlens.io import read_articles_excel, read_competition_labeled_excel
from eventlens.tapt import TaptConfig, run_tapt_mlm
from eventlens.transformer_event import TransformerTrainingConfig, run_transformer_event_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled-limit", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--tapt-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    unlabeled = read_articles_excel(
        settings.paths.untagged_train,
        nrows=args.unlabeled_limit,
        task_scope="company_event",
    )
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]

    tapt_config = TaptConfig(
        model=args.base_model or settings.transformer_event.model,
        max_length=256,
        max_content_chars=settings.transformer_event.max_content_chars,
        batch_size=16,
        max_steps=args.max_steps,
        learning_rate=5e-5,
        random_state=settings.model.random_state,
    )
    tapt_report = run_tapt_mlm(
        unlabeled,
        config=tapt_config,
        output_dir=args.tapt_dir,
        cache_dir="/root/autodl-tmp/hf_cache",
        local_files_only=True,
        device="cuda",
    )

    finetune_config = TransformerTrainingConfig(
        model=args.tapt_dir,
        max_length=settings.transformer_event.max_length,
        max_content_chars=settings.transformer_event.max_content_chars,
        batch_size=settings.transformer_event.batch_size,
        epochs=settings.transformer_event.epochs,
        learning_rate=settings.transformer_event.learning_rate,
        weight_decay=settings.transformer_event.weight_decay,
        validation_ratio=settings.transformer_event.validation_ratio,
        warmup_ratio=settings.transformer_event.warmup_ratio,
        label_smoothing=settings.transformer_event.label_smoothing,
        class_weight_power=settings.transformer_event.class_weight_power,
        early_stopping_patience=settings.transformer_event.early_stopping_patience,
        include_subject_fields=False,
        gate_macro_f1=0.80,
        random_state=settings.model.random_state,
    )
    finetune_report = run_transformer_event_experiment(
        train,
        test,
        scope="company",
        config=finetune_config,
        baseline_external_macro_f1=0.770798,
        cache_dir="/root/autodl-tmp/hf_cache",
        local_files_only=True,
        device="cuda",
    )
    payload = {
        "scope": "company",
        "protocol": "production-like TAPT pilot on target-domain unlabeled news, then identical supervised MacBERT recipe; no labeled-only subject fields",
        "tapt": tapt_report,
        "finetune": finetune_report.model_dump(),
        "supervised_macbert_reference_macro_f1": 0.541564,
        "production_like_svc_reference_macro_f1": 0.770798,
        "tapt_gain_vs_supervised_macbert": round(
            finetune_report.external.macro_f1 - 0.541564, 6
        ),
        "gap_vs_production_like_svc": round(
            finetune_report.external.macro_f1 - 0.770798, 6
        ),
        "gate_macro_f1": 0.80,
        "gate_passed": finetune_report.external.macro_f1 >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
