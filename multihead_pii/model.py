from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from .labels import BIO_LABELS, SENSITIVITY_LABELS, TYPE_LABELS
from .losses import combine_multitask_losses, masked_cross_entropy, masked_soft_cross_entropy


class MultiHeadPiiModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_span_len: int,
        span_width_vocab_size: int = 64,
        dropout: float = 0.1,
        proposal_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        sensitivity_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.max_span_len = max_span_len
        self.loss_weights = {
            "proposal": proposal_loss_weight,
            "type": type_loss_weight,
            "sensitivity": sensitivity_loss_weight,
        }

        self.dropout = nn.Dropout(dropout)
        self.proposal_head = nn.Linear(hidden, len(BIO_LABELS))
        self.width_emb = nn.Embedding(span_width_vocab_size, hidden // 2)

        span_dim = hidden * 3 + (hidden // 2)
        self.type_head = nn.Sequential(
            nn.Linear(span_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(TYPE_LABELS)),
        )
        self.sensitivity_head = nn.Sequential(
            nn.Linear(span_dim + len(TYPE_LABELS), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(SENSITIVITY_LABELS)),
        )

    def _span_representations(
        self,
        hidden: torch.Tensor,
        candidate_spans: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_reps = []
        batch_meta = []
        device = hidden.device

        for batch_idx in range(hidden.size(0)):
            spans = candidate_spans[batch_idx]
            valid_mask = spans[:, 0] >= 0
            valid_spans = spans[valid_mask]
            if valid_spans.numel() == 0:
                continue
            starts = valid_spans[:, 0]
            ends = valid_spans[:, 1]
            start_h = hidden[batch_idx, starts]
            end_h = hidden[batch_idx, ends]

            pooled = []
            for s, e in zip(starts.tolist(), ends.tolist()):
                pooled.append(hidden[batch_idx, s : e + 1].mean(dim=0))
            pooled_h = torch.stack(pooled, dim=0)

            widths = (ends - starts + 1).clamp(max=self.width_emb.num_embeddings - 1)
            width_h = self.width_emb(widths.to(device))

            rep = torch.cat([start_h, end_h, pooled_h, width_h], dim=-1)
            batch_reps.append(rep)
            for local_idx in range(valid_spans.size(0)):
                batch_meta.append((batch_idx, int(valid_mask.nonzero()[local_idx].item())))

        if not batch_reps:
            return {
                "span_repr": torch.empty(0, 1, device=hidden.device),
                "meta": [],
            }
        return {"span_repr": torch.cat(batch_reps, dim=0), "meta": batch_meta}

    def _gather_flat_labels(
        self,
        labels: Optional[torch.Tensor],
        meta: list,
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        out = []
        for batch_idx, span_idx in meta:
            out.append(labels[batch_idx, span_idx])
        if not out:
            return torch.empty(0, dtype=torch.long, device=labels.device)
        return torch.stack(out)

    def _gather_flat_soft_labels(
        self,
        labels: Optional[torch.Tensor],
        meta: list,
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        out = []
        for batch_idx, span_idx in meta:
            out.append(labels[batch_idx, span_idx])
        if not out:
            return torch.empty(0, labels.size(-1), dtype=labels.dtype, device=labels.device)
        return torch.stack(out)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_spans: torch.Tensor,
        proposal_labels: Optional[torch.Tensor] = None,
        type_labels: Optional[torch.Tensor] = None,
        sensitivity_labels: Optional[torch.Tensor] = None,
        type_soft_labels: Optional[torch.Tensor] = None,
        sensitivity_soft_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        proposal_logits = self.proposal_head(self.dropout(hidden))

        span_bundle = self._span_representations(hidden, candidate_spans)
        span_repr = span_bundle["span_repr"]
        meta = span_bundle["meta"]

        type_logits = None
        sensitivity_logits = None
        flat_type_labels = None
        flat_sensitivity_labels = None
        flat_type_soft_labels = None
        flat_sensitivity_soft_labels = None
        if span_repr.numel() > 0:
            type_logits = self.type_head(self.dropout(span_repr))
            type_probs = torch.softmax(type_logits, dim=-1)
            sensitivity_inputs = torch.cat([span_repr, type_probs], dim=-1)
            sensitivity_logits = self.sensitivity_head(self.dropout(sensitivity_inputs))
            flat_type_labels = self._gather_flat_labels(type_labels, meta)
            flat_sensitivity_labels = self._gather_flat_labels(sensitivity_labels, meta)
            flat_type_soft_labels = self._gather_flat_soft_labels(type_soft_labels, meta)
            flat_sensitivity_soft_labels = self._gather_flat_soft_labels(sensitivity_soft_labels, meta)

        proposal_loss = masked_cross_entropy(
            proposal_logits.view(-1, proposal_logits.size(-1)),
            proposal_labels.view(-1) if proposal_labels is not None else None,
            ignore_index=-100,
        ).to(hidden.device)
        if flat_type_soft_labels is not None:
            type_loss = masked_soft_cross_entropy(
                type_logits,
                flat_type_soft_labels,
            ).to(hidden.device)
        else:
            type_loss = masked_cross_entropy(
                type_logits,
                flat_type_labels,
                ignore_index=-100,
            ).to(hidden.device)
        if flat_sensitivity_soft_labels is not None:
            sensitivity_loss = masked_soft_cross_entropy(
                sensitivity_logits,
                flat_sensitivity_soft_labels,
            ).to(hidden.device)
        else:
            sensitivity_loss = masked_cross_entropy(
                sensitivity_logits,
                flat_sensitivity_labels,
                ignore_index=-100,
            ).to(hidden.device)

        total_loss = combine_multitask_losses(
            proposal_loss=proposal_loss,
            type_loss=type_loss,
            sensitivity_loss=sensitivity_loss,
            weights=self.loss_weights,
        )

        return {
            "loss": total_loss,
            "proposal_loss": proposal_loss,
            "type_loss": type_loss,
            "sensitivity_loss": sensitivity_loss,
            "proposal_logits": proposal_logits,
            "type_logits": type_logits,
            "sensitivity_logits": sensitivity_logits,
            "meta": meta,
        }


def load_checkpoint(checkpoint_path: str, device: str) -> Dict:
    """Load a raw checkpoint dict from disk."""
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(checkpoint: Dict, device: str) -> "MultiHeadPiiModel":
    """Instantiate and return an eval-mode MultiHeadPiiModel from a checkpoint dict."""
    from .config import MultiHeadConfig
    cfg = MultiHeadConfig(**checkpoint["config"])
    model = MultiHeadPiiModel(
        model_name=cfg.model_name,
        max_span_len=cfg.max_span_len,
        span_width_vocab_size=cfg.span_width_vocab_size,
        dropout=cfg.dropout,
        proposal_loss_weight=cfg.proposal_loss_weight,
        type_loss_weight=cfg.type_loss_weight,
        sensitivity_loss_weight=cfg.sensitivity_loss_weight,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model
