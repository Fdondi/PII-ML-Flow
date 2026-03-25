import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import MultiHeadConfig
from .dataset import JsonlMultiHeadDataset, _extract_value, collate_fn
from .decoder import (
    DecodedSpan,
    attach_regex_candidates,
    decode_final_spans,
    non_max_suppression,
    select_non_overlapping_typed_spans,
)
from .labels import TYPE_ID_TO_LABEL
from .model import MultiHeadPiiModel, build_model_from_checkpoint, load_checkpoint
from .type_comparison import ValueKey, classify_value_relationships, make_value_key
from .train import resolve_device


def _value_dict(label: str, value: str) -> Dict:
    return {
        "value": value,
        "label": label,
    }


def _enumerate_candidates_from_offsets(
    offsets: List[Tuple[int, int]],
    max_span_len: int,
) -> List[Tuple[int, int]]:
    valid_idxs = [i for i, off in enumerate(offsets) if off != (0, 0)]
    candidates: List[Tuple[int, int]] = []
    for start in valid_idxs:
        max_end = min(start + max_span_len - 1, len(offsets) - 1)
        for end in range(start, max_end + 1):
            if offsets[end] == (0, 0):
                break
            candidates.append((start, end))
    return candidates


def _infer_window_spans(
    text: str,
    offsets: List[Tuple[int, int]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    bundle: Dict[str, Any],
) -> List[DecodedSpan]:
    config: MultiHeadConfig = bundle["config"]
    model: MultiHeadPiiModel = bundle["model"]
    device: str = bundle["device"]

    candidate_spans = _enumerate_candidates_from_offsets(
        offsets=offsets,
        max_span_len=config.max_span_len,
    )
    if config.include_regex_candidates:
        candidate_spans = attach_regex_candidates(
            text=text,
            offsets=offsets,
            candidate_spans=candidate_spans,
            max_span_len=config.max_span_len,
        )
    if not candidate_spans:
        return []

    candidate_tensor = torch.tensor(candidate_spans, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        candidate_spans=candidate_tensor,
    )
    type_logits = outputs["type_logits"]
    sens_logits = outputs["sensitivity_logits"]
    if type_logits is None or sens_logits is None:
        return []

    return decode_final_spans(
        text=text,
        offsets=offsets,
        candidate_spans=candidate_tensor[0],
        type_logits=type_logits,
        sensitivity_logits=sens_logits,
        redact_score_threshold=config.redact_score_threshold,
        nms_iou_threshold=config.nms_iou_threshold,
    )


def _merge_window_redactions(spans: List[DecodedSpan], iou_threshold: float) -> List[DecodedSpan]:
    best_by_key: Dict[Tuple[int, int, str], DecodedSpan] = {}
    for span in spans:
        key = (span.start, span.end, span.label)
        prev = best_by_key.get(key)
        if prev is None or span.redact_score > prev.redact_score:
            best_by_key[key] = span
    deduped = list(best_by_key.values())
    return non_max_suppression(deduped, iou_threshold=iou_threshold)


def load_inference_bundle(checkpoint_path: str, device: str = "auto") -> Dict[str, Any]:
    resolved_device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, device=resolved_device)
    config = MultiHeadConfig(**checkpoint["config"])

    encoder_source = str((Path(checkpoint_path).parent / "encoder").resolve())
    if not Path(encoder_source).exists():
        encoder_source = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)
    model = build_model_from_checkpoint(checkpoint, device=resolved_device)
    return {
        "device": resolved_device,
        "config": config,
        "tokenizer": tokenizer,
        "model": model,
    }


@torch.no_grad()
def infer_text(text: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    config: MultiHeadConfig = bundle["config"]
    tokenizer = bundle["tokenizer"]
    device: str = bundle["device"]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=config.max_length,
        return_overflowing_tokens=True,
        stride=max(1, config.max_length // 8),
        return_offsets_mapping=True,
    )
    input_batches = enc["input_ids"]
    attention_batches = enc["attention_mask"]
    offset_batches = enc["offset_mapping"]

    all_decoded: List[DecodedSpan] = []
    for i in range(len(input_batches)):
        offsets = [tuple(x) for x in offset_batches[i]]
        input_ids_tensor = torch.tensor(input_batches[i], dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(attention_batches[i], dtype=torch.long, device=device)
        decoded = _infer_window_spans(
            text=text,
            offsets=offsets,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            bundle=bundle,
        )
        all_decoded.extend(decoded)

    merged = _merge_window_redactions(
        all_decoded,
        iou_threshold=config.nms_iou_threshold,
    )
    return {
        "text": text,
        "truncated": False,
        "windowed": len(input_batches) > 1,
        "window_count": len(input_batches),
        "redactions": [
            {
                "start": span.start,
                "end": span.end,
                "value": _extract_value(text, span.start, span.end),
                "label": span.label,
                "decision": span.decision,
                "redact_score": span.redact_score,
                "redact_probability": span.redact_probability,
                "type_confidence": span.type_confidence,
            }
            for span in merged
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-head PII inference.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output", default="outputs_multihead/predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sensitivity", default=None, help="Optional sensitivity companion JSONL.")
    parser.add_argument(
        "--pretty-output",
        default=None,
        help=(
            "Optional pretty JSON output path. Defaults to <output stem>.pretty.jsonl "
            "when not provided."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    config = MultiHeadConfig(**checkpoint["config"])

    encoder_source = str((Path(args.checkpoint).parent / "encoder").resolve())
    if not (Path(encoder_source).exists()):
        encoder_source = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(encoder_source, use_fast=True)

    dataset = JsonlMultiHeadDataset(
        path=args.input,
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model_from_checkpoint(checkpoint, device=device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for batch in loader:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        outputs = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            candidate_spans=batch_tensors["candidate_spans"],
        )

        type_logits = outputs["type_logits"]
        sens_logits = outputs["sensitivity_logits"]
        cursor = 0
        for i, raw in enumerate(batch["raw"]):
            spans_i = batch_tensors["candidate_spans"][i]
            valid_mask = spans_i[:, 0] >= 0
            n_valid = int(valid_mask.sum().item())
            spans_valid = spans_i[valid_mask]
            if n_valid == 0 or type_logits is None or sens_logits is None:
                all_rows.append(
                    {
                        "text": raw["text"],
                        "redactions": [],
                        "type_comparison": {
                            "true_positives": [],
                            "false_positives": [],
                            "false_negatives": [],
                        },
                    }
                )
                continue

            type_i = type_logits[cursor : cursor + n_valid]
            sens_i = sens_logits[cursor : cursor + n_valid]
            cursor += n_valid

            final_redact = decode_final_spans(
                text=raw["text"],
                offsets=raw["offsets"],
                candidate_spans=spans_valid,
                type_logits=type_i,
                sensitivity_logits=sens_i,
                redact_score_threshold=config.redact_score_threshold,
                nms_iou_threshold=config.nms_iou_threshold,
            )

            type_probs = torch.softmax(type_i, dim=-1)
            pred_type_candidates: List[Tuple[int, int, str, float]] = []
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

            # Remove nested/overlapping predictions to avoid contradictory TP/FP accounting.
            pred_type_set: Set[Tuple[int, int, str]] = set(
                select_non_overlapping_typed_spans(pred_type_candidates)
            )

            pred_type_value_keys: Set[ValueKey] = set()
            for start, end, label in pred_type_set:
                value = _extract_value(raw["text"], start, end)
                pred_type_value_keys.add(make_value_key(label, value))

            gold_type_value_keys: Set[ValueKey] = set()
            for span in raw.get("gold_type_spans", []):
                label = span.get("label")
                value = span.get("value")
                start = span.get("start")
                end = span.get("end")
                if not isinstance(label, str) or label == "NONE":
                    continue
                if isinstance(value, str):
                    gold_type_value_keys.add(make_value_key(label, value))
                    continue
                if isinstance(start, int) and isinstance(end, int):
                    gold_value = _extract_value(raw["text"], start, end)
                    gold_type_value_keys.add(make_value_key(label, gold_value))

            comparison = classify_value_relationships(
                pred_keys=pred_type_value_keys,
                gold_keys=gold_type_value_keys,
            )

            all_rows.append(
                {
                    "text": raw["text"],
                    "redactions": [
                        {
                            "start": span.start,
                            "end": span.end,
                            "value": _extract_value(raw["text"], span.start, span.end),
                            "label": span.label,
                            "decision": span.decision,
                            "redact_score": span.redact_score,
                            "redact_probability": span.redact_probability,
                            "type_confidence": span.type_confidence,
                        }
                        for span in final_redact
                    ],
                    "type_comparison": {
                        "true_positives": [
                            _value_dict(label=row["label"], value=row["value"])
                            for row in comparison["exact_tp"]
                        ],
                        "false_positives": [
                            _value_dict(label=row["label"], value=row["value"])
                            for row in comparison["exact_fp"]
                        ],
                        "false_negatives": [
                            _value_dict(label=row["label"], value=row["value"])
                            for row in comparison["exact_fn"]
                        ],
                        "type_mismatches": comparison["type_mismatches"],
                        "value_subsets": comparison["value_subsets"],
                        "value_supersets": comparison["value_supersets"],
                    },
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=True, indent=2) + "\n")
    print(f"saved predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
