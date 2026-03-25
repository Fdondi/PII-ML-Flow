import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import MultiHeadConfig
from .dataset import JsonlMultiHeadDataset, _extract_value, char_span_to_token_span, collate_fn
from .decoder import decode_final_spans, select_non_overlapping_typed_spans
from .labels import SENSITIVITY_LABEL_TO_ID, TYPE_ID_TO_LABEL
from .model import MultiHeadPiiModel, build_model_from_checkpoint, load_checkpoint
from .span_credit import soft_match_total_credit
from .type_comparison import ValueKey, classify_value_relationships, make_value_key
from .train import resolve_device


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-head PII model.")
    parser.add_argument("--valid", required=True, help="Validation JSONL path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output", default="outputs_multihead/eval_report.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sensitivity", default=None, help="Optional sensitivity companion JSONL.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    config = MultiHeadConfig(**checkpoint["config"])
    model = build_model_from_checkpoint(checkpoint, device=device)

    encoder_source = str((Path(args.checkpoint).parent / "encoder").resolve())
    if not Path(encoder_source).exists():
        encoder_source = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)

    dataset = JsonlMultiHeadDataset(
        path=args.valid,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_span_len=config.max_span_len,
        negative_sample_rate=1.0,
        training=False,
        sensitivity_path=args.sensitivity,
        include_regex_candidates=config.include_regex_candidates,
        lookalike_redact_target=config.lookalike_redact_target,
        no_info_keep_target=config.no_info_keep_target,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    losses = {"loss": 0.0, "proposal_loss": 0.0, "type_loss": 0.0, "sensitivity_loss": 0.0}
    total_batches = 0

    proposal_correct = 0
    proposal_total = 0
    sens_correct = 0
    sens_total = 0
    sens_mae_sum = 0.0
    sens_brier_sum = 0.0

    type_tp = type_fp = type_fn = 0
    type_mismatch = 0
    type_subset = 0
    type_superset = 0
    type_soft_tp = 0.0
    type_soft_pred_total = 0
    type_soft_gold_total = 0
    redact_tp = redact_fp = redact_fn = 0
    redact_soft_tp = 0.0
    redact_soft_pred_total = 0
    redact_soft_gold_total = 0
    docs = 0

    for batch in loader:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        outputs = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            candidate_spans=batch_tensors["candidate_spans"],
            proposal_labels=batch_tensors["proposal_labels"],
            type_labels=batch_tensors["type_labels"],
            sensitivity_labels=batch_tensors["sensitivity_labels"],
            type_soft_labels=batch_tensors.get("type_soft_labels"),
            sensitivity_soft_labels=batch_tensors.get("sensitivity_soft_labels"),
        )
        for key in losses:
            losses[key] += float(outputs[key].item())
        total_batches += 1

        proposal_logits = outputs["proposal_logits"]
        proposal_labels = batch_tensors["proposal_labels"]
        proposal_pred = proposal_logits.argmax(dim=-1)
        valid_tokens = proposal_labels != -100
        proposal_correct += int((proposal_pred[valid_tokens] == proposal_labels[valid_tokens]).sum().item())
        proposal_total += int(valid_tokens.sum().item())

        type_logits = outputs["type_logits"]
        sensitivity_logits = outputs["sensitivity_logits"]
        cursor = 0
        for i, raw in enumerate(batch["raw"]):
            docs += 1
            spans_i = batch_tensors["candidate_spans"][i]
            valid_mask = spans_i[:, 0] >= 0
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0 or type_logits is None or sensitivity_logits is None:
                continue

            spans_valid = spans_i[valid_mask]
            type_i = type_logits[cursor : cursor + n_valid]
            sens_i = sensitivity_logits[cursor : cursor + n_valid]
            type_labels_i = batch_tensors["type_labels"][i][valid_mask]
            sens_labels_i = batch_tensors["sensitivity_labels"][i][valid_mask]
            cursor += n_valid

            sens_pred = sens_i.argmax(dim=-1)
            sens_mask = sens_labels_i != -100
            sens_correct += int((sens_pred[sens_mask] == sens_labels_i[sens_mask]).sum().item())
            sens_total += int(sens_mask.sum().item())
            if int(sens_mask.sum().item()) > 0:
                sens_probs = torch.softmax(sens_i, dim=-1)
                redact_probs = sens_probs[:, SENSITIVITY_LABEL_TO_ID["REDACT"]]
                targets = (sens_labels_i == SENSITIVITY_LABEL_TO_ID["REDACT"]).float()
                masked_preds = redact_probs[sens_mask]
                masked_targets = targets[sens_mask]
                sens_mae_sum += float(torch.abs(masked_preds - masked_targets).sum().item())
                sens_brier_sum += float(((masked_preds - masked_targets) ** 2).sum().item())

            pred_type_candidates: List[Tuple[int, int, str, float]] = []
            type_probs = torch.softmax(type_i, dim=-1)
            for j in range(n_valid):
                span = spans_valid[j]
                token_start = int(span[0].item())
                token_end = int(span[1].item())
                char_start = raw["offsets"][token_start][0]
                char_end = raw["offsets"][token_end][1]
                if char_end <= char_start:
                    continue
                type_id = int(type_probs[j].argmax(dim=-1).item())
                label = TYPE_ID_TO_LABEL[type_id]
                if label == "NONE":
                    continue
                score = float(type_probs[j, type_id].item())
                pred_type_candidates.append((char_start, char_end, label, score))

            pred_type_set: Set[Tuple[int, int, str]] = set(
                select_non_overlapping_typed_spans(pred_type_candidates)
            )
            pred_type_tok: List[Tuple[int, int, str]] = []
            for start, end, label in pred_type_set:
                tok_span = char_span_to_token_span(raw["offsets"], start, end)
                if tok_span is None:
                    continue
                pred_type_tok.append((tok_span[0], tok_span[1], label))

            pred_type_value_keys: Set[ValueKey] = set()
            for start, end, label in pred_type_set:
                value = _extract_value(raw["text"], start, end)
                pred_type_value_keys.add(make_value_key(label, value))

            gold_type_value_keys: Set[ValueKey] = set()
            gold_type_tok: List[Tuple[int, int, str]] = []
            for span in raw.get("gold_type_spans", []):
                label = span.get("label")
                value = span.get("value")
                start = span.get("start")
                end = span.get("end")
                if isinstance(label, str) and isinstance(value, str):
                    gold_type_value_keys.add(make_value_key(label, value))
                elif isinstance(start, int) and isinstance(end, int) and isinstance(label, str):
                    gold_value = _extract_value(raw["text"], start, end)
                    gold_type_value_keys.add(make_value_key(label, gold_value))
                if isinstance(start, int) and isinstance(end, int) and isinstance(label, str):
                    tok_span = char_span_to_token_span(raw["offsets"], start, end)
                    if tok_span is not None:
                        gold_type_tok.append((tok_span[0], tok_span[1], label))

            comparison = classify_value_relationships(
                pred_keys=pred_type_value_keys,
                gold_keys=gold_type_value_keys,
            )
            type_tp += len(comparison["exact_tp"])
            type_fp += len(comparison["exact_fp"])
            type_fn += len(comparison["exact_fn"])
            type_mismatch += len(comparison["type_mismatches"])
            type_subset += len(comparison["value_subsets"])
            type_superset += len(comparison["value_supersets"])
            type_soft_tp += soft_match_total_credit(
                pred_spans=pred_type_tok,
                gold_spans=gold_type_tok,
                require_same_label=True,
            )
            type_soft_pred_total += len(pred_type_tok)
            type_soft_gold_total += len(gold_type_tok)

            final_redact = decode_final_spans(
                text=raw["text"],
                offsets=raw["offsets"],
                candidate_spans=spans_valid,
                type_logits=type_i,
                sensitivity_logits=sens_i,
                redact_score_threshold=config.redact_score_threshold,
                nms_iou_threshold=config.nms_iou_threshold,
            )
            pred_redact_set = {(x.start, x.end) for x in final_redact}
            gold_redact_set = {
                (int(x["start"]), int(x["end"]))
                for x in raw.get("gold_redact_spans", [])
                if isinstance(x, dict) and isinstance(x.get("start"), int) and isinstance(x.get("end"), int)
            }
            redact_tp += len(pred_redact_set & gold_redact_set)
            redact_fp += len(pred_redact_set - gold_redact_set)
            redact_fn += len(gold_redact_set - pred_redact_set)
            pred_redact_tok: List[Tuple[int, int, str]] = []
            for start, end in pred_redact_set:
                tok_span = char_span_to_token_span(raw["offsets"], start, end)
                if tok_span is None:
                    continue
                pred_redact_tok.append((tok_span[0], tok_span[1], "REDACT"))
            gold_redact_tok: List[Tuple[int, int, str]] = []
            for start, end in gold_redact_set:
                tok_span = char_span_to_token_span(raw["offsets"], start, end)
                if tok_span is None:
                    continue
                gold_redact_tok.append((tok_span[0], tok_span[1], "REDACT"))
            redact_soft_tp += soft_match_total_credit(
                pred_spans=pred_redact_tok,
                gold_spans=gold_redact_tok,
                require_same_label=False,
            )
            redact_soft_pred_total += len(pred_redact_tok)
            redact_soft_gold_total += len(gold_redact_tok)

    report = {
        "num_docs": docs,
        "avg_losses": {k: _safe_div(v, total_batches) for k, v in losses.items()},
        "proposal_accuracy": _safe_div(proposal_correct, proposal_total),
        "sensitivity_candidate_accuracy": _safe_div(sens_correct, sens_total),
        "sensitivity_redact_probability_mae": _safe_div(sens_mae_sum, sens_total),
        "sensitivity_redact_probability_brier": _safe_div(sens_brier_sum, sens_total),
        "type_metrics": _prf(type_tp, type_fp, type_fn),
        "type_overlap_metrics": _prf(
            type_soft_tp,
            float(type_soft_pred_total) - type_soft_tp,
            float(type_soft_gold_total) - type_soft_tp,
        ),
        "redaction_metrics": _prf(redact_tp, redact_fp, redact_fn),
        "redaction_overlap_metrics": _prf(
            redact_soft_tp,
            float(redact_soft_pred_total) - redact_soft_tp,
            float(redact_soft_gold_total) - redact_soft_tp,
        ),
        "span_counts": {
            "type": {"tp": type_tp, "fp": type_fp, "fn": type_fn},
            "type_overlap_credit": {
                "tp_credit": type_soft_tp,
                "predicted_total": type_soft_pred_total,
                "gold_total": type_soft_gold_total,
            },
            "type_non_exact_breakdown": {
                "type_mismatch": type_mismatch,
                "value_subset": type_subset,
                "value_superset": type_superset,
            },
            "redaction": {"tp": redact_tp, "fp": redact_fp, "fn": redact_fn},
            "redaction_overlap_credit": {
                "tp_credit": redact_soft_tp,
                "predicted_total": redact_soft_pred_total,
                "gold_total": redact_soft_gold_total,
            },
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
