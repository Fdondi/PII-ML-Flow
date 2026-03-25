import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pii_labels import TRAINING_LABELS
from multihead_pii.dataset import LABEL_ALIASES

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 512
MAX_SPAN_LEN = 6
NEGATIVE_SAMPLE_RATE = 0.20
BATCH_SIZE = 4
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SpanExample:
    text: str
    spans: List[Dict]


class JsonlSpanDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = MAX_LENGTH,
        max_span_len: int = MAX_SPAN_LEN,
        negative_sample_rate: float = NEGATIVE_SAMPLE_RATE,
    ):
        self.examples = self._load_jsonl(path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.max_span_len = max_span_len
        self.negative_sample_rate = negative_sample_rate

    @staticmethod
    def _normalize_label(raw_label: object) -> Optional[str]:
        if not isinstance(raw_label, str):
            return None
        label = raw_label.strip().upper()
        if not label:
            return None
        label = LABEL_ALIASES.get(label, label)
        if label in TRAINING_LABELS:
            return label
        return "OTHER"

    @classmethod
    def _normalize_span(cls, span: object, text: str) -> Optional[Dict]:
        if not isinstance(span, dict):
            return None
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            start = span.get("start_position")
            end = span.get("end_position")
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        label = cls._normalize_label(span.get("label"))
        if label is None:
            label = cls._normalize_label(span.get("entity_type"))
        if label is None:
            return None
        value = span.get("value")
        if not isinstance(value, str):
            value = span.get("entity_value")
        if not isinstance(value, str):
            safe_start = max(0, min(start, len(text)))
            safe_end = max(safe_start, min(end, len(text)))
            value = text[safe_start:safe_end]
        return {"start": start, "end": end, "label": label, "value": value}

    @classmethod
    def _load_jsonl(cls, path: str) -> List[SpanExample]:
        file_path = Path(path)
        rows: List[Dict] = []
        if file_path.suffix.lower() == ".json":
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                rows = [raw]
            elif isinstance(raw, list):
                if not all(isinstance(x, dict) for x in raw):
                    raise ValueError(f"{path}: JSON array must contain only objects")
                rows = raw
            else:
                raise ValueError(f"{path}: JSON must be object or array of objects")
        else:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}: each JSONL row must be an object")
                    rows.append(row)

        data = []
        for idx, row in enumerate(rows):
            text = row.get("text")
            if not isinstance(text, str):
                text = row.get("full_text")
            if not isinstance(text, str):
                raise ValueError(f"row {idx}: missing/invalid text or full_text")
            raw_spans = row.get("spans")
            if raw_spans is None:
                raw_spans = []
            if not isinstance(raw_spans, list):
                raise ValueError(f"row {idx}: spans must be a list")
            spans: List[Dict] = []
            for span in raw_spans:
                norm = cls._normalize_span(span, text)
                if norm is not None:
                    spans.append(norm)
            data.append(SpanExample(text=text, spans=spans))
        return data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
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
        offsets = enc["offset_mapping"][0].tolist()

        candidate_spans, labels = build_span_candidates(
            offsets=offsets,
            gold_spans=ex.spans,
            label2id=self.label2id,
            max_span_len=self.max_span_len,
            negative_sample_rate=self.negative_sample_rate,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_spans": torch.tensor(candidate_spans, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": ex.text,
            "offsets": offsets,
        }


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
    for i, (s, e) in enumerate(offsets):
        if not is_valid_token((s, e)):
            continue
        if overlap(s, e, char_start, char_end):
            if token_start is None:
                token_start = i
            token_end = i
    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def build_span_candidates(
    offsets: List[Tuple[int, int]],
    gold_spans: List[Dict],
    label2id: Dict[str, int],
    max_span_len: int,
    negative_sample_rate: float,
):
    gold_map = {}
    for span in gold_spans:
        tok_span = char_span_to_token_span(offsets, span["start"], span["end"])
        if tok_span is None:
            continue
        label = span["label"]
        if label not in label2id:
            continue
        gold_map[tok_span] = label2id[label]

    valid_idxs = [i for i, off in enumerate(offsets) if is_valid_token(off)]
    if not valid_idxs:
        return [[0, 0]], [label2id["NONE"]]

    candidates = []
    labels = []
    for start in valid_idxs:
        for end in range(start, min(start + max_span_len, len(offsets))):
            if not is_valid_token(offsets[end]):
                break
            label = gold_map.get((start, end), label2id["NONE"])
            if label == label2id["NONE"] and random.random() > negative_sample_rate:
                continue
            candidates.append([start, end])
            labels.append(label)

    if not candidates:
        candidates = [[valid_idxs[0], valid_idxs[0]]]
        labels = [label2id["NONE"]]
    return candidates, labels


class SpanClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, max_span_len: int = MAX_SPAN_LEN, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.length_emb = nn.Embedding(max_span_len + 1, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask, candidate_spans, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, T, H]

        batch_reps = []
        batch_labels = []
        for b in range(hidden.size(0)):
            spans = candidate_spans[b]
            valid_mask = spans[:, 0] >= 0
            spans = spans[valid_mask]
            if spans.numel() == 0:
                continue
            starts = spans[:, 0]
            ends = spans[:, 1]
            start_h = hidden[b, starts]
            end_h = hidden[b, ends]

            pooled = []
            for s, e in zip(starts.tolist(), ends.tolist()):
                pooled.append(hidden[b, s : e + 1].mean(dim=0))
            pooled = torch.stack(pooled, dim=0)
            lengths = (ends - starts + 1).clamp(max=self.length_emb.num_embeddings - 1)
            len_h = self.length_emb(lengths)

            rep = torch.cat([start_h, end_h, pooled, len_h], dim=-1)
            batch_reps.append(rep)
            if labels is not None:
                batch_labels.append(labels[b][valid_mask])

        reps = torch.cat(batch_reps, dim=0)
        logits = self.classifier(self.dropout(reps))

        loss = None
        if labels is not None:
            flat_labels = torch.cat(batch_labels, dim=0)
            loss = nn.CrossEntropyLoss()(logits, flat_labels)
        return {"loss": loss, "logits": logits}


def collate_fn(batch):
    max_toks = max(x["input_ids"].shape[0] for x in batch)
    max_spans = max(x["candidate_spans"].shape[0] for x in batch)

    input_ids = []
    attention_mask = []
    candidate_spans = []
    labels = []

    for x in batch:
        pad_t = max_toks - x["input_ids"].shape[0]
        input_ids.append(torch.nn.functional.pad(x["input_ids"], (0, pad_t), value=0))
        attention_mask.append(torch.nn.functional.pad(x["attention_mask"], (0, pad_t), value=0))

        pad_s = max_spans - x["candidate_spans"].shape[0]
        candidate_spans.append(torch.nn.functional.pad(x["candidate_spans"], (0, 0, 0, pad_s), value=-1))
        labels.append(torch.nn.functional.pad(x["labels"], (0, pad_s), value=-100))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "candidate_spans": torch.stack(candidate_spans),
        "labels": torch.stack(labels),
        "raw": batch,
    }


def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            candidate_spans=batch["candidate_spans"],
            labels=batch["labels"],
        )
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            candidate_spans=batch["candidate_spans"],
            labels=batch["labels"],
        )
        total_loss += outputs["loss"].item()
    return total_loss / max(1, len(loader))


def main(train_path="train.jsonl", valid_path="valid.jsonl", output_dir="outputs"):
    set_seed()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = TRAINING_LABELS
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for x, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = JsonlSpanDataset(train_path, tokenizer, label2id)
    valid_ds = JsonlSpanDataset(valid_path, tokenizer, label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SpanClassifier(MODEL_NAME, num_labels=len(labels)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=max(1, total_steps),
    )

    best_valid = math.inf
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        valid_loss = evaluate_loss(model, valid_loader)
        print(f"epoch={epoch+1} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            model.encoder.save_pretrained(output_dir / "encoder")
            tokenizer.save_pretrained(output_dir / "encoder")
            torch.save(
                {
                    "classifier": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "max_span_len": MAX_SPAN_LEN,
                },
                output_dir / "span_classifier.pt",
            )

    print(f"Saved best model to {output_dir.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.jsonl")
    parser.add_argument("--valid", default="valid.jsonl")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    main(train_path=args.train, valid_path=args.valid, output_dir=args.output)
