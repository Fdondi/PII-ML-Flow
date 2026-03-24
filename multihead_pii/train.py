import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import MultiHeadConfig, load_config
from .dataset import JsonlMultiHeadDataset, collate_fn
from .labels import (
    BIO_LABEL_TO_ID,
    SENSITIVITY_LABEL_TO_ID,
    TYPE_LABEL_TO_ID,
)
from .model import MultiHeadPiiModel

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def train_one_epoch(
    model: MultiHeadPiiModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "proposal_loss": 0.0, "type_loss": 0.0, "sensitivity_loss": 0.0}
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            candidate_spans=batch["candidate_spans"],
            proposal_labels=batch["proposal_labels"],
            type_labels=batch["type_labels"],
            sensitivity_labels=batch["sensitivity_labels"],
            type_soft_labels=batch.get("type_soft_labels"),
            sensitivity_soft_labels=batch.get("sensitivity_soft_labels"),
        )
        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()
        scheduler.step()
        for key in totals:
            totals[key] += float(outputs[key].item())
    denom = max(1, len(loader))
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def evaluate_loss(
    model: MultiHeadPiiModel,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "proposal_loss": 0.0, "type_loss": 0.0, "sensitivity_loss": 0.0}
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            candidate_spans=batch["candidate_spans"],
            proposal_labels=batch["proposal_labels"],
            type_labels=batch["type_labels"],
            sensitivity_labels=batch["sensitivity_labels"],
            type_soft_labels=batch.get("type_soft_labels"),
            sensitivity_soft_labels=batch.get("sensitivity_soft_labels"),
        )
        for key in totals:
            totals[key] += float(outputs[key].item())
    denom = max(1, len(loader))
    return {k: v / denom for k, v in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-head PII model.")
    parser.add_argument("--train", required=True, help="Training JSONL path.")
    parser.add_argument("--valid", required=True, help="Validation JSONL path.")
    parser.add_argument(
        "--config",
        default="configs/multihead_v1.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--output",
        default="outputs_multihead",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--train-sensitivity",
        default=None,
        help="Optional JSONL sensitivity companion for training rows.",
    )
    parser.add_argument(
        "--valid-sensitivity",
        default=None,
        help="Optional JSONL sensitivity companion for validation rows.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="pii-multihead",
        help="MLflow experiment name (default: pii-multihead).",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="MLflow run name. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking even if mlflow is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: MultiHeadConfig = load_config(args.config)
    set_seed(config.seed)
    device = resolve_device(config.device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_mlflow = _MLFLOW_AVAILABLE and not args.no_mlflow
    if use_mlflow:
        mlflow.set_experiment(args.mlflow_experiment)
        active_run = mlflow.start_run(run_name=args.mlflow_run_name)
        # Log all hyperparameters
        mlflow.log_params({
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "dropout": config.dropout,
            "epochs": config.epochs,
            "train_batch_size": config.train_batch_size,
            "max_span_len": config.max_span_len,
            "negative_sample_rate": config.negative_sample_rate,
            "proposal_loss_weight": config.proposal_loss_weight,
            "type_loss_weight": config.type_loss_weight,
            "sensitivity_loss_weight": config.sensitivity_loss_weight,
            "lookalike_redact_target": config.lookalike_redact_target,
            "no_info_keep_target": config.no_info_keep_target,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "seed": config.seed,
            "max_length": config.max_length,
            "nms_iou_threshold": config.nms_iou_threshold,
            "redact_score_threshold": config.redact_score_threshold,
        })

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    train_ds = JsonlMultiHeadDataset(
        path=args.train,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_span_len=config.max_span_len,
        negative_sample_rate=config.negative_sample_rate,
        training=True,
        sensitivity_path=args.train_sensitivity,
        include_regex_candidates=config.include_regex_candidates,
        lookalike_redact_target=config.lookalike_redact_target,
        no_info_keep_target=config.no_info_keep_target,
    )
    valid_ds = JsonlMultiHeadDataset(
        path=args.valid,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_span_len=config.max_span_len,
        negative_sample_rate=1.0,
        training=False,
        sensitivity_path=args.valid_sensitivity,
        include_regex_candidates=config.include_regex_candidates,
        lookalike_redact_target=config.lookalike_redact_target,
        no_info_keep_target=config.no_info_keep_target,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = MultiHeadPiiModel(
        model_name=config.model_name,
        max_span_len=config.max_span_len,
        span_width_vocab_size=config.span_width_vocab_size,
        dropout=config.dropout,
        proposal_loss_weight=config.proposal_loss_weight,
        type_loss_weight=config.type_loss_weight,
        sensitivity_loss_weight=config.sensitivity_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_valid = float("inf")
    monitored_losses = ("proposal_loss", "type_loss", "sensitivity_loss")
    best_monitored = {name: float("inf") for name in monitored_losses}
    epochs_without_progress = 0
    history = []
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        valid_metrics = evaluate_loss(model, valid_loader, device)
        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f}"
        )

        if use_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_proposal_loss": train_metrics["proposal_loss"],
                    "train_type_loss": train_metrics["type_loss"],
                    "train_sensitivity_loss": train_metrics["sensitivity_loss"],
                    "valid_loss": valid_metrics["loss"],
                    "valid_proposal_loss": valid_metrics["proposal_loss"],
                    "valid_type_loss": valid_metrics["type_loss"],
                    "valid_sensitivity_loss": valid_metrics["sensitivity_loss"],
                },
                step=epoch + 1,
            )

        progressed_losses = []
        min_delta = max(0.0, float(config.early_stopping_min_delta))
        for loss_name in monitored_losses:
            loss_value = valid_metrics[loss_name]
            if math.isfinite(loss_value) and loss_value < (best_monitored[loss_name] - min_delta):
                best_monitored[loss_name] = loss_value
                progressed_losses.append(loss_name)

        if progressed_losses:
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1

        print(
            "early-stop monitor: "
            f"no_progress={epochs_without_progress}/{config.early_stopping_patience}, "
            f"min_delta={min_delta:.6f}, "
            f"best_proposal={best_monitored['proposal_loss']:.6f}, "
            f"best_type={best_monitored['type_loss']:.6f}, "
            f"best_sensitivity={best_monitored['sensitivity_loss']:.6f}"
        )

        valid_loss_value = valid_metrics["loss"]
        is_better = (
            epoch == 0
            or (math.isfinite(valid_loss_value) and valid_loss_value < best_valid)
        )
        if is_better:
            best_valid = valid_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "type_label_to_id": TYPE_LABEL_TO_ID,
                    "sensitivity_label_to_id": SENSITIVITY_LABEL_TO_ID,
                    "bio_label_to_id": BIO_LABEL_TO_ID,
                },
                output_dir / "multihead_model.pt",
            )
            model.encoder.save_pretrained(output_dir / "encoder")
            tokenizer.save_pretrained(output_dir / "encoder")

        if config.early_stopping_patience > 0 and epochs_without_progress >= config.early_stopping_patience:
            print(
                f"early stopping at epoch {epoch + 1} "
                f"(no significant progress in {config.early_stopping_patience} epochs "
                f"across proposal/type/sensitivity validation losses)"
            )
            break

    history_path = output_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if use_mlflow:
        mlflow.log_metric("best_valid_loss", best_valid)
        mlflow.log_metric("best_valid_proposal_loss", best_monitored["proposal_loss"])
        mlflow.log_metric("best_valid_type_loss", best_monitored["type_loss"])
        mlflow.log_metric("best_valid_sensitivity_loss", best_monitored["sensitivity_loss"])
        mlflow.log_metric("epochs_trained", len(history))
        mlflow.log_artifact(str(history_path), artifact_path="training")
        mlflow.log_artifact(str(output_dir / "multihead_model.pt"), artifact_path="model")
        mlflow.end_run()

    print(f"saved best checkpoint to {(output_dir / 'multihead_model.pt').resolve()}")
    print(f"saved train history to {history_path.resolve()}")


if __name__ == "__main__":
    main()
