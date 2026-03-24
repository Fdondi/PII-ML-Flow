#!/usr/bin/env python3
"""Hyperparameter search for MultiHeadPiiModel using MLflow tracking.

Runs random search over the most impactful hyperparameters and tracks
every trial in an MLflow experiment.  After all trials finish, prints
a summary table ranked by best validation loss.

Usage
-----
    python hparam_search.py \\
        --train  train.google-gemma-3-27b.jsonl \\
        --valid  valid.google-gemma-3-27b.jsonl \\
        --base-config configs/multihead_v1.json \\
        --n-trials 20 \\
        --experiment pii-hparam-search \\
        --output-root hparam_outputs

The script writes one subdirectory per trial inside --output-root and
passes --mlflow-experiment / --mlflow-run-name to the training process.

Search space (see SEARCH_SPACE below)
--------------------------------------
Tier 1 – transformer fine-tuning fundamentals
  learning_rate          log-uniform [1e-5, 5e-5]
  dropout                uniform     [0.05, 0.30]
  weight_decay           log-uniform [1e-3, 0.10]

Tier 2 – training dynamics
  warmup_ratio           uniform     [0.05, 0.20]
  negative_sample_rate   uniform     [0.10, 0.50]

Tier 3 – multitask loss balance
  proposal_loss_weight   uniform     [0.5, 2.0]
  type_loss_weight       uniform     [0.5, 2.0]
  sensitivity_loss_weight uniform    [0.5, 2.0]

Tier 4 – span-candidate architecture
  max_span_len           choice      [8, 10, 12, 16]
"""

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from multihead_pii.config import load_config, save_config

# ---------------------------------------------------------------------------
# Search space definition
# Each entry is (kind, *args):
#   ("log_uniform", lo, hi)  – sample exp(uniform(log(lo), log(hi)))
#   ("uniform",     lo, hi)  – sample uniform(lo, hi)
#   ("choice",      [v, …])  – pick one element at random
# ---------------------------------------------------------------------------
SEARCH_SPACE: Dict[str, Tuple] = {
    # Tier 1
    "learning_rate":           ("log_uniform", 1e-5,  5e-5),
    "dropout":                 ("uniform",     0.05,  0.30),
    "weight_decay":            ("log_uniform", 1e-3,  0.10),
    # Tier 2
    "warmup_ratio":            ("uniform",     0.05,  0.20),
    "negative_sample_rate":    ("uniform",     0.10,  0.50),
    # Tier 3
    "proposal_loss_weight":    ("uniform",     0.5,   2.0),
    "type_loss_weight":        ("uniform",     0.5,   2.0),
    "sensitivity_loss_weight": ("uniform",     0.5,   2.0),
    # Tier 4
    "max_span_len":            ("choice",      [8, 10, 12, 16]),
}


def _sample(spec: Tuple, rng: random.Random) -> Any:
    kind = spec[0]
    if kind == "uniform":
        return rng.uniform(spec[1], spec[2])
    if kind == "log_uniform":
        lo, hi = math.log(spec[1]), math.log(spec[2])
        return math.exp(rng.uniform(lo, hi))
    if kind == "choice":
        return rng.choice(spec[1])
    raise ValueError(f"Unknown sampling kind: {kind!r}")


def sample_hparams(space: Dict[str, Tuple], rng: random.Random) -> Dict[str, Any]:
    """Draw one set of hyperparameters from the search space."""
    return {name: _sample(spec, rng) for name, spec in space.items()}


def _round_hparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Round floats to a sensible number of significant figures."""
    result = {}
    for k, v in hparams.items():
        if isinstance(v, float):
            # 4 sig-figs is enough for all params here
            result[k] = float(f"{v:.4g}")
        else:
            result[k] = v
    return result


def build_trial_config(base_config_path: str, hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Merge sampled hyperparameters into the base config dict."""
    cfg = load_config(base_config_path)
    cfg_dict = asdict(cfg)
    cfg_dict.update(hparams)
    return cfg_dict


def run_trial(
    trial_idx: int,
    hparams: Dict[str, Any],
    args: argparse.Namespace,
    tmp_dir: Path,
) -> Dict[str, Any]:
    """Write a temp config, launch training, return a result summary dict."""
    run_name = f"trial-{trial_idx:03d}"
    trial_output = Path(args.output_root) / run_name
    trial_output.mkdir(parents=True, exist_ok=True)

    # Build config for this trial
    cfg_dict = build_trial_config(args.base_config, hparams)
    cfg_path = tmp_dir / f"{run_name}_config.json"
    cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "multihead_pii.train",
        "--train",  args.train,
        "--valid",  args.valid,
        "--config", str(cfg_path),
        "--output", str(trial_output),
        "--mlflow-experiment", args.experiment,
        "--mlflow-run-name",   run_name,
    ]
    if args.train_sensitivity:
        cmd += ["--train-sensitivity", args.train_sensitivity]
    if args.valid_sensitivity:
        cmd += ["--valid-sensitivity", args.valid_sensitivity]

    print(f"\n{'='*60}")
    print(f"[hparam_search] Starting {run_name}")
    print(f"  hparams: {json.dumps(hparams, indent=4)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    # Read best valid loss from train_history if available
    best_valid_loss = float("inf")
    history_path = trial_output / "train_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                best_valid_loss = min(row["valid"]["loss"] for row in history)
        except Exception:
            pass

    return {
        "trial": run_name,
        "returncode": result.returncode,
        "best_valid_loss": best_valid_loss,
        "hparams": hparams,
        "output_dir": str(trial_output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random hyperparameter search for MultiHeadPiiModel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",  required=True, help="Training JSONL path.")
    parser.add_argument("--valid",  required=True, help="Validation JSONL path.")
    parser.add_argument(
        "--base-config", default="configs/multihead_v1.json",
        help="Base config JSON to override with sampled hyperparameters.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Number of random trials to run.",
    )
    parser.add_argument(
        "--experiment", default="pii-hparam-search",
        help="MLflow experiment name shared by all trials.",
    )
    parser.add_argument(
        "--output-root", default="hparam_outputs",
        help="Root directory; one subdirectory is created per trial.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the hyperparameter sampler (not the model seed).",
    )
    parser.add_argument(
        "--train-sensitivity", default=None,
        help="Optional JSONL sensitivity companion for training rows.",
    )
    parser.add_argument(
        "--valid-sensitivity", default=None,
        help="Optional JSONL sensitivity companion for validation rows.",
    )
    return parser.parse_args()


def print_summary(results: List[Dict[str, Any]]) -> None:
    sorted_results = sorted(results, key=lambda r: r["best_valid_loss"])
    col_w = 14
    header = (
        f"{'Trial':<12} {'best_val_loss':>{col_w}} "
        f"{'lr':>{col_w}} {'dropout':>{col_w}} {'wd':>{col_w}} "
        f"{'warmup':>{col_w}} {'neg_rate':>{col_w}} "
        f"{'prop_w':>{col_w}} {'type_w':>{col_w}} {'sens_w':>{col_w}} "
        f"{'span_len':>{col_w}}"
    )
    print("\n" + "="*len(header))
    print("HYPERPARAMETER SEARCH SUMMARY (sorted by best_val_loss)")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    for r in sorted_results:
        hp = r["hparams"]
        ok = "OK" if r["returncode"] == 0 else "ERR"
        loss_str = f"{r['best_valid_loss']:.6f}" if r["best_valid_loss"] < float("inf") else "N/A"
        print(
            f"{r['trial']:<12} {loss_str:>{col_w}} "
            f"{hp.get('learning_rate', '?'):>{col_w}.2e} "
            f"{hp.get('dropout', '?'):>{col_w}.3f} "
            f"{hp.get('weight_decay', '?'):>{col_w}.4f} "
            f"{hp.get('warmup_ratio', '?'):>{col_w}.3f} "
            f"{hp.get('negative_sample_rate', '?'):>{col_w}.3f} "
            f"{hp.get('proposal_loss_weight', '?'):>{col_w}.3f} "
            f"{hp.get('type_loss_weight', '?'):>{col_w}.3f} "
            f"{hp.get('sensitivity_loss_weight', '?'):>{col_w}.3f} "
            f"{hp.get('max_span_len', '?'):>{col_w}}  {ok}"
        )
    print("="*len(header))
    if sorted_results:
        best = sorted_results[0]
        print(f"\nBest trial : {best['trial']}  (valid_loss={best['best_valid_loss']:.6f})")
        print(f"Output dir : {best['output_dir']}")
        print(f"Hparams    : {json.dumps(best['hparams'], indent=4)}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(args.n_trials):
            hparams = _round_hparams(sample_hparams(SEARCH_SPACE, rng))
            result = run_trial(i, hparams, args, Path(tmp_dir))
            all_results.append(result)
            # Checkpoint progress after each trial
            summary_path = Path(args.output_root) / "search_results.json"
            summary_path.write_text(
                json.dumps(all_results, indent=2), encoding="utf-8"
            )

    print_summary(all_results)
    summary_path = Path(args.output_root) / "search_results.json"
    print(f"\nFull results saved to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
