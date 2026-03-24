import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from .decoder import (
    CREDIT_CARD_PATTERN,
    EMAIL_PATTERN,
    IBAN_PATTERN,
    IPV4_PATTERN,
    PHONE_PATTERN,
)
from .labels import (
    BIO_LABEL_TO_ID,
    SENSITIVITY_LABEL_TO_ID,
    TYPE_LABEL_TO_ID,
    sensitivity_from_item_category,
)
from .span_credit import overlap_credit

REGEX_TYPE_MAP = {
    "EMAIL": EMAIL_PATTERN,
    "PHONE": PHONE_PATTERN,
    "IP_ADDRESS": IPV4_PATTERN,
    "IBAN": IBAN_PATTERN,
    "CREDIT_CARD": CREDIT_CARD_PATTERN,
}

LABEL_ALIASES = {
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "ORGANIZATION": "ORG",
    "IP": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "DATE_TIME": "OTHER",
    "LOCATION": "OTHER",
    "CITY": "OTHER",
    "COUNTRY": "OTHER",
    "STATE": "OTHER",
    "ZIPCODE": "ADDRESS",
    "ZIP_CODE": "ADDRESS",
    "POSTAL_CODE": "ADDRESS",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "NAME": "PERSON",
}


@dataclass
class MultiHeadExample:
    text: str
    spans: List[Dict]
    items: List[Dict]
    sensitivity_spans: List[Dict]


def is_valid_token(offset_pair: Tuple[int, int]) -> bool:
    start, end = offset_pair
    return not (start == 0 and end == 0)


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def char_span_to_token_span(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Optional[Tuple[int, int]]:
    token_start = None
    token_end = None
    for i, (start, end) in enumerate(offsets):
        if not is_valid_token((start, end)):
            continue
        if overlap(start, end, char_start, char_end):
            if token_start is None:
                token_start = i
            token_end = i
    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def token_span_to_char_span(
    offsets: List[Tuple[int, int]],
    token_start: int,
    token_end: int,
) -> Optional[Tuple[int, int]]:
    if token_start < 0 or token_end < token_start or token_end >= len(offsets):
        return None
    start_off = offsets[token_start]
    end_off = offsets[token_end]
    if not is_valid_token(start_off) or not is_valid_token(end_off):
        return None
    char_start = start_off[0]
    char_end = end_off[1]
    if char_end <= char_start:
        return None
    return char_start, char_end


def _span_key(start: int, end: int) -> Tuple[int, int]:
    return (start, end)


def _extract_value(text: str, start: int, end: int) -> str:
    safe_start = max(0, min(start, len(text)))
    safe_end = max(safe_start, min(end, len(text)))
    return text[safe_start:safe_end]


def _normalize_label(raw_label: object) -> Optional[str]:
    if not isinstance(raw_label, str):
        return None
    label = raw_label.strip().upper()
    if not label:
        return None
    label = LABEL_ALIASES.get(label, label)
    if label in TYPE_LABEL_TO_ID:
        return label
    return "OTHER"


def _normalize_span_dict(span: object, text: str) -> Optional[Dict]:
    if not isinstance(span, dict):
        return None

    start = span.get("start")
    end = span.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        start = span.get("start_position")
        end = span.get("end_position")
    if not isinstance(start, int) or not isinstance(end, int):
        return None

    out: Dict = {"start": start, "end": end}
    label = _normalize_label(span.get("label"))
    if label is None:
        label = _normalize_label(span.get("entity_type"))
    if label is not None:
        out["label"] = label

    sensitivity = span.get("sensitivity")
    if isinstance(sensitivity, str):
        sensitivity = sensitivity.strip().upper()
        if sensitivity in SENSITIVITY_LABEL_TO_ID:
            out["sensitivity"] = sensitivity

    category = span.get("category")
    if isinstance(category, str):
        out["category"] = category

    value = span.get("value")
    if not isinstance(value, str):
        value = span.get("entity_value")
    if not isinstance(value, str):
        value = _extract_value(text, start, end)
    out["value"] = value
    return out


def _normalize_span_list(spans: object, text: str) -> List[Dict]:
    if spans is None:
        return []
    if not isinstance(spans, list):
        raise ValueError("spans/items/sensitivity_spans must be a list")
    out: List[Dict] = []
    for span in spans:
        norm = _normalize_span_dict(span, text)
        if norm is not None:
            out.append(norm)
    return out


def _collect_gold_type_spans(example: MultiHeadExample) -> List[Dict]:
    out: Dict[Tuple[int, int, str], Dict] = {}

    # Prefer richer item annotations first.
    for item in example.items:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        label = item.get("label")
        category = item.get("category")
        if category == "NON_PII_NUMBER":
            continue
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(label, str)
            or label not in TYPE_LABEL_TO_ID
            or label == "NONE"
        ):
            continue
        value = item.get("value")
        if not isinstance(value, str):
            value = _extract_value(example.text, start, end)
        key = (start, end, label)
        out[key] = {"start": start, "end": end, "label": label, "value": value}

    for span in example.spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        label = span.get("label")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(label, str)
            or label not in TYPE_LABEL_TO_ID
            or label == "NONE"
        ):
            continue
        value = span.get("value")
        if not isinstance(value, str):
            value = _extract_value(example.text, start, end)
        key = (start, end, label)
        out.setdefault(key, {"start": start, "end": end, "label": label, "value": value})

    return sorted(out.values(), key=lambda row: (row["start"], row["end"], row["label"]))


def _collect_gold_redact_spans(example: MultiHeadExample) -> List[Dict]:
    out: Dict[Tuple[int, int], Dict] = {}

    for item in example.items:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        category = item.get("category")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        sensitivity = sensitivity_from_item_category(category)
        if sensitivity != "REDACT":
            continue
        value = item.get("value")
        if not isinstance(value, str):
            value = _extract_value(example.text, start, end)
        out[(start, end)] = {"start": start, "end": end, "value": value}

    for span in example.sensitivity_spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        sensitivity = span.get("sensitivity")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(sensitivity, str)
            or sensitivity != "REDACT"
        ):
            continue
        value = span.get("value")
        if not isinstance(value, str):
            value = _extract_value(example.text, start, end)
        out.setdefault((start, end), {"start": start, "end": end, "value": value})

    return sorted(out.values(), key=lambda row: (row["start"], row["end"]))


def _load_rows(path: str) -> List[Dict]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            if not all(isinstance(x, dict) for x in raw):
                raise ValueError(f"{path}: JSON array must contain only objects")
            return raw
        raise ValueError(f"{path}: JSON must be object or array of objects")

    if suffix != ".jsonl":
        raise ValueError(f"{path}: unsupported extension (expected .json or .jsonl)")

    rows: List[Dict] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no}: row is not an object")
            rows.append(row)
    return rows


def _normalize_rows(
    main_rows: List[Dict],
    sensitivity_rows: Optional[List[Dict]] = None,
) -> List[MultiHeadExample]:
    sensitivity_rows = sensitivity_rows or []
    sensitivity_by_index = {i: row for i, row in enumerate(sensitivity_rows)}
    examples: List[MultiHeadExample] = []
    for idx, row in enumerate(main_rows):
        text = row.get("text")
        if not isinstance(text, str):
            text = row.get("full_text")
        if not isinstance(text, str):
            raise ValueError(f"row {idx}: missing/invalid text or full_text")
        spans = _normalize_span_list(row.get("spans"), text)
        items = _normalize_span_list(row.get("items"), text)
        companion = sensitivity_by_index.get(idx, {})
        companion_spans = []
        if isinstance(companion, dict):
            companion_spans = _normalize_span_list(companion.get("sensitivity_spans", []), text)
        examples.append(
            MultiHeadExample(
                text=text,
                spans=spans,
                items=items,
                sensitivity_spans=companion_spans,
            )
        )
    return examples


def _regex_char_candidates(text: str) -> List[Tuple[int, int]]:
    found: Set[Tuple[int, int]] = set()
    for pattern in REGEX_TYPE_MAP.values():
        for match in pattern.finditer(text):
            found.add((match.start(), match.end()))
    return sorted(found)


def _build_supervision_maps(
    example: MultiHeadExample,
) -> Tuple[
    Dict[Tuple[int, int], str],
    Dict[Tuple[int, int], str],
    Set[Tuple[int, int]],
    Set[Tuple[int, int]],
]:
    type_by_char_span: Dict[Tuple[int, int], str] = {}
    sens_by_char_span: Dict[Tuple[int, int], str] = {}
    non_pii_spans: Set[Tuple[int, int]] = set()
    lookalike_spans: Set[Tuple[int, int]] = set()

    # Prefer richer item annotations when present.
    if example.items:
        for item in example.items:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            label = item.get("label")
            category = item.get("category")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            key = _span_key(start, end)
            if category == "NON_PII_NUMBER":
                non_pii_spans.add(key)
                continue
            if isinstance(category, str) and category.upper() == "PII_LOOKALIKE":
                lookalike_spans.add(key)
            if isinstance(label, str) and label in TYPE_LABEL_TO_ID:
                type_by_char_span[key] = label
            sensitivity = sensitivity_from_item_category(category)
            if sensitivity is not None:
                sens_by_char_span[key] = sensitivity

    # Span-only annotations still train type and proposal.
    for span in example.spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        label = span.get("label")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(label, str)
            or label not in TYPE_LABEL_TO_ID
            or label == "NONE"
        ):
            continue
        key = _span_key(start, end)
        type_by_char_span.setdefault(key, label)

    # Companion file gives explicit sensitivity labels for span-only datasets.
    for span in example.sensitivity_spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        sensitivity = span.get("sensitivity")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(sensitivity, str)
            or sensitivity not in SENSITIVITY_LABEL_TO_ID
        ):
            continue
        sens_by_char_span[_span_key(start, end)] = sensitivity

    # Span-only datasets have no explicit sensitivity labels.
    # In that case, default typed spans to REDACT so sensitivity head is trained.
    if not sens_by_char_span and type_by_char_span:
        for key in type_by_char_span:
            sens_by_char_span[key] = "REDACT"

    return type_by_char_span, sens_by_char_span, non_pii_spans, lookalike_spans


def _build_proposal_labels(
    offsets: List[Tuple[int, int]],
    positive_tok_spans: Set[Tuple[int, int]],
) -> List[int]:
    labels = [-100 for _ in offsets]
    for i, off in enumerate(offsets):
        if is_valid_token(off):
            labels[i] = BIO_LABEL_TO_ID["O"]
    for start, end in positive_tok_spans:
        if start < 0 or end >= len(labels) or end < start:
            continue
        labels[start] = BIO_LABEL_TO_ID["B-ENTITY"]
        for idx in range(start + 1, end + 1):
            labels[idx] = BIO_LABEL_TO_ID["I-ENTITY"]
    return labels


def _enumerate_candidates(
    offsets: List[Tuple[int, int]],
    max_span_len: int,
) -> List[Tuple[int, int]]:
    valid_idxs = [i for i, off in enumerate(offsets) if is_valid_token(off)]
    candidates: List[Tuple[int, int]] = []
    for start in valid_idxs:
        max_end = min(start + max_span_len - 1, len(offsets) - 1)
        for end in range(start, max_end + 1):
            if not is_valid_token(offsets[end]):
                break
            candidates.append((start, end))
    return candidates


def _build_type_soft_target(
    candidate: Tuple[int, int],
    exact_type_id: int,
    gold_type_spans: List[Tuple[Tuple[int, int], int]],
) -> List[float]:
    n_types = len(TYPE_LABEL_TO_ID)
    none_id = TYPE_LABEL_TO_ID["NONE"]
    target = [0.0] * n_types

    if exact_type_id != none_id:
        target[exact_type_id] = 1.0
        return target

    def iou_with_overlapping_gold_group(
        candidate_span: Tuple[int, int],
        gold_spans_for_label: List[Tuple[int, int]],
    ) -> float:
        pred_start, pred_end = candidate_span
        if pred_end < pred_start:
            return 0.0
        overlapping_gold = [
            span
            for span in gold_spans_for_label
            if overlap_credit(candidate_span, span) > 0.0
        ]
        if not overlapping_gold:
            return 0.0
        pred_tokens = set(range(pred_start, pred_end + 1))
        gold_tokens = set()
        for gold_start, gold_end in overlapping_gold:
            if gold_end >= gold_start:
                gold_tokens.update(range(gold_start, gold_end + 1))
        if not pred_tokens or not gold_tokens:
            return 0.0
        union = pred_tokens | gold_tokens
        if not union:
            return 0.0
        intersection = pred_tokens & gold_tokens
        return float(len(intersection)) / float(len(union))

    by_type: Dict[int, List[Tuple[int, int]]] = {}
    for gold_span, gold_type_id in gold_type_spans:
        if gold_type_id == none_id:
            continue
        by_type.setdefault(gold_type_id, []).append(gold_span)

    best_score = 0.0
    best_type_id = none_id
    for gold_type_id, spans in by_type.items():
        score = iou_with_overlapping_gold_group(candidate, spans)
        if score > best_score:
            best_score = score
            best_type_id = gold_type_id

    if best_score > 0.0 and best_type_id != none_id:
        target[best_type_id] = best_score
        target[none_id] = 1.0 - best_score
    else:
        target[none_id] = 1.0
    return target


def _build_sensitivity_soft_target(
    candidate: Tuple[int, int],
    exact_sens_id: int,
    gold_sens_spans: List[Tuple[Tuple[int, int], int]],
    exact_is_lookalike: bool = False,
    lookalike_redact_target: float = 0.8,
    no_info_keep_target: float = 1.0,
) -> List[float]:
    n_sens = len(SENSITIVITY_LABEL_TO_ID)
    target = [0.0] * n_sens
    redact_id = SENSITIVITY_LABEL_TO_ID.get("REDACT")
    keep_id = SENSITIVITY_LABEL_TO_ID.get("KEEP")

    if exact_is_lookalike:
        # Bias lookalikes toward REDACT to prefer caution.
        redact_mass = max(0.0, min(1.0, float(lookalike_redact_target)))
        if redact_id is not None:
            target[redact_id] = redact_mass
        if keep_id is not None:
            target[keep_id] = 1.0 - redact_mass
        elif redact_id is None and n_sens > 0:
            target[0] = 1.0
        return target

    if exact_sens_id != -100:
        target[exact_sens_id] = 1.0
        return target

    def iou_with_overlapping_gold_group(
        candidate_span: Tuple[int, int],
        gold_spans_for_label: List[Tuple[int, int]],
    ) -> float:
        pred_start, pred_end = candidate_span
        if pred_end < pred_start:
            return 0.0
        overlapping_gold = [
            span
            for span in gold_spans_for_label
            if overlap_credit(candidate_span, span) > 0.0
        ]
        if not overlapping_gold:
            return 0.0
        pred_tokens = set(range(pred_start, pred_end + 1))
        gold_tokens = set()
        for gold_start, gold_end in overlapping_gold:
            if gold_end >= gold_start:
                gold_tokens.update(range(gold_start, gold_end + 1))
        if not pred_tokens or not gold_tokens:
            return 0.0
        union = pred_tokens | gold_tokens
        if not union:
            return 0.0
        intersection = pred_tokens & gold_tokens
        return float(len(intersection)) / float(len(union))

    by_sens: Dict[int, List[Tuple[int, int]]] = {}
    for gold_span, gold_sens_id in gold_sens_spans:
        if gold_sens_id < 0:
            continue
        by_sens.setdefault(gold_sens_id, []).append(gold_span)

    best_score = 0.0
    best_sens_id = -100
    for gold_sens_id, spans in by_sens.items():
        score = iou_with_overlapping_gold_group(candidate, spans)
        if score > best_score:
            best_score = score
            best_sens_id = gold_sens_id

    if best_score <= 0.0 or best_sens_id == -100:
        keep_mass = max(0.0, min(1.0, float(no_info_keep_target)))
        if keep_id is not None:
            target[keep_id] = keep_mass
            remaining = 1.0 - keep_mass
            if redact_id is not None:
                target[redact_id] = remaining
            elif n_sens > 1:
                spread = remaining / float(n_sens - 1)
                for i in range(n_sens):
                    if i != keep_id:
                        target[i] = spread
        elif n_sens > 0:
            target[0] = 1.0
        return target

    # Spread remaining mass to the opposite class.
    # Sensitivity currently has exactly two classes.
    target[best_sens_id] = best_score
    if n_sens == 2:
        other = 1 - best_sens_id
        target[other] = 1.0 - best_score
    else:
        remainder = (1.0 - best_score) / max(1, n_sens - 1)
        for i in range(n_sens):
            if i != best_sens_id:
                target[i] = remainder
    return target


class JsonlMultiHeadDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        max_span_len: int,
        negative_sample_rate: float = 0.2,
        training: bool = True,
        sensitivity_path: Optional[str] = None,
        include_regex_candidates: bool = True,
        lookalike_redact_target: float = 0.8,
        no_info_keep_target: float = 1.0,
    ):
        main_rows = _load_rows(path)
        sensitivity_rows = _load_rows(sensitivity_path) if sensitivity_path else None
        if sensitivity_rows and len(sensitivity_rows) != len(main_rows):
            raise ValueError(
                "Sensitivity companion row count mismatch: "
                f"{len(sensitivity_rows)} vs {len(main_rows)}"
            )
        self.examples = _normalize_rows(main_rows, sensitivity_rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_span_len = max_span_len
        self.negative_sample_rate = negative_sample_rate
        self.training = training
        self.include_regex_candidates = include_regex_candidates
        self.lookalike_redact_target = lookalike_redact_target
        self.no_info_keep_target = no_info_keep_target

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]

        type_by_char, sens_by_char, _, lookalike_by_char = _build_supervision_maps(ex)
        type_by_tok: Dict[Tuple[int, int], int] = {}
        sens_by_tok: Dict[Tuple[int, int], int] = {}
        lookalike_by_tok: Set[Tuple[int, int]] = set()
        positive_tok_spans: Set[Tuple[int, int]] = set()

        for (start, end), label in type_by_char.items():
            tok_span = char_span_to_token_span(offsets, start, end)
            if tok_span is None:
                continue
            type_by_tok[tok_span] = TYPE_LABEL_TO_ID[label]
            positive_tok_spans.add(tok_span)

        for (start, end), sensitivity in sens_by_char.items():
            tok_span = char_span_to_token_span(offsets, start, end)
            if tok_span is None:
                continue
            sens_by_tok[tok_span] = SENSITIVITY_LABEL_TO_ID[sensitivity]

        for start, end in lookalike_by_char:
            tok_span = char_span_to_token_span(offsets, start, end)
            if tok_span is None:
                continue
            lookalike_by_tok.add(tok_span)

        proposal_labels = _build_proposal_labels(offsets, positive_tok_spans)
        all_candidates = _enumerate_candidates(offsets, max_span_len=self.max_span_len)

        candidate_spans: List[List[int]] = []
        type_labels: List[int] = []
        sensitivity_labels: List[int] = []
        type_soft_labels: List[List[float]] = []
        sensitivity_soft_labels: List[List[float]] = []

        positive_set = set(type_by_tok.keys()) | set(sens_by_tok.keys())
        gold_type_spans = [(span, label_id) for span, label_id in type_by_tok.items()]
        gold_sens_spans = [(span, label_id) for span, label_id in sens_by_tok.items()]
        for candidate in all_candidates:
            is_positive = candidate in positive_set
            if self.training and not is_positive and random.random() > self.negative_sample_rate:
                continue
            candidate_spans.append([candidate[0], candidate[1]])
            type_label = type_by_tok.get(candidate, TYPE_LABEL_TO_ID["NONE"])
            sens_label = sens_by_tok.get(candidate, -100)
            type_labels.append(type_label)
            sensitivity_labels.append(sens_label)
            type_soft_labels.append(
                _build_type_soft_target(
                    candidate=candidate,
                    exact_type_id=type_label,
                    gold_type_spans=gold_type_spans,
                )
            )
            sensitivity_soft_labels.append(
                _build_sensitivity_soft_target(
                    candidate=candidate,
                    exact_sens_id=sens_label,
                    gold_sens_spans=gold_sens_spans,
                    exact_is_lookalike=candidate in lookalike_by_tok,
                    lookalike_redact_target=self.lookalike_redact_target,
                    no_info_keep_target=self.no_info_keep_target,
                )
            )

        if self.include_regex_candidates:
            existing = {tuple(x) for x in candidate_spans}
            for char_start, char_end in _regex_char_candidates(ex.text):
                tok_span = char_span_to_token_span(offsets, char_start, char_end)
                if tok_span is None or tok_span in existing:
                    continue
                if tok_span[1] - tok_span[0] + 1 > self.max_span_len:
                    continue
                candidate_spans.append([tok_span[0], tok_span[1]])
                type_label = type_by_tok.get(tok_span, TYPE_LABEL_TO_ID["NONE"])
                sens_label = sens_by_tok.get(tok_span, -100)
                type_labels.append(type_label)
                sensitivity_labels.append(sens_label)
                type_soft_labels.append(
                    _build_type_soft_target(
                        candidate=tok_span,
                        exact_type_id=type_label,
                        gold_type_spans=gold_type_spans,
                    )
                )
                sensitivity_soft_labels.append(
                    _build_sensitivity_soft_target(
                        candidate=tok_span,
                        exact_sens_id=sens_label,
                        gold_sens_spans=gold_sens_spans,
                        exact_is_lookalike=tok_span in lookalike_by_tok,
                        lookalike_redact_target=self.lookalike_redact_target,
                        no_info_keep_target=self.no_info_keep_target,
                    )
                )
                existing.add(tok_span)

        if not candidate_spans:
            # Keep shape-valid fallback.
            candidate_spans = [[0, 0]]
            type_labels = [TYPE_LABEL_TO_ID["NONE"]]
            sensitivity_labels = [-100]
            type_soft_labels = [[1.0] + [0.0] * (len(TYPE_LABEL_TO_ID) - 1)]
            sensitivity_soft_labels = [
                _build_sensitivity_soft_target(
                    candidate=(0, 0),
                    exact_sens_id=-100,
                    gold_sens_spans=[],
                    exact_is_lookalike=False,
                    lookalike_redact_target=self.lookalike_redact_target,
                    no_info_keep_target=self.no_info_keep_target,
                )
            ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_spans": torch.tensor(candidate_spans, dtype=torch.long),
            "proposal_labels": torch.tensor(proposal_labels, dtype=torch.long),
            "type_labels": torch.tensor(type_labels, dtype=torch.long),
            "sensitivity_labels": torch.tensor(sensitivity_labels, dtype=torch.long),
            "type_soft_labels": torch.tensor(type_soft_labels, dtype=torch.float),
            "sensitivity_soft_labels": torch.tensor(sensitivity_soft_labels, dtype=torch.float),
            "raw": {
                "text": ex.text,
                "offsets": offsets,
                "gold_type_spans": _collect_gold_type_spans(ex),
                "gold_redact_spans": _collect_gold_redact_spans(ex),
            },
        }


def collate_fn(batch: List[Dict]) -> Dict:
    max_toks = max(x["input_ids"].shape[0] for x in batch)
    max_spans = max(x["candidate_spans"].shape[0] for x in batch)

    input_ids = []
    attention_mask = []
    candidate_spans = []
    proposal_labels = []
    type_labels = []
    sensitivity_labels = []
    type_soft_labels = []
    sensitivity_soft_labels = []

    for item in batch:
        pad_t = max_toks - item["input_ids"].shape[0]
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_t), value=0))
        attention_mask.append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_t), value=0)
        )
        proposal_labels.append(
            torch.nn.functional.pad(item["proposal_labels"], (0, pad_t), value=-100)
        )

        pad_s = max_spans - item["candidate_spans"].shape[0]
        candidate_spans.append(
            torch.nn.functional.pad(item["candidate_spans"], (0, 0, 0, pad_s), value=-1)
        )
        type_labels.append(torch.nn.functional.pad(item["type_labels"], (0, pad_s), value=-100))
        sensitivity_labels.append(
            torch.nn.functional.pad(item["sensitivity_labels"], (0, pad_s), value=-100)
        )
        type_soft_labels.append(
            torch.nn.functional.pad(
                item["type_soft_labels"],
                (0, 0, 0, pad_s),
                value=0.0,
            )
        )
        sensitivity_soft_labels.append(
            torch.nn.functional.pad(
                item["sensitivity_soft_labels"],
                (0, 0, 0, pad_s),
                value=-1.0,
            )
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "candidate_spans": torch.stack(candidate_spans),
        "proposal_labels": torch.stack(proposal_labels),
        "type_labels": torch.stack(type_labels),
        "sensitivity_labels": torch.stack(sensitivity_labels),
        "type_soft_labels": torch.stack(type_soft_labels),
        "sensitivity_soft_labels": torch.stack(sensitivity_soft_labels),
        "raw": [item["raw"] for item in batch],
    }
