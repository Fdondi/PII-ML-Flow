#!/usr/bin/env python3
"""Hyperparameter search for MultiHeadPiiModel using Optuna + MLflow.

Uses Optuna's TPE sampler (Tree-structured Parzen Estimator) for Bayesian
optimisation: each trial informs the next, converging faster than random search.
Every trial is logged as a separate MLflow run inside --experiment.

When MLflow 3.x is available the Optuna study is persisted to the MLflow
Tracking Server via MlflowStorage, so searches can be paused and resumed
and the full optimisation trajectory is visible in the MLflow UI.

Usage
-----
    python hparam_search.py \\
        --train  train.google-gemma-3-27b.jsonl \\
        --valid  valid.google-gemma-3-27b.jsonl \\
        --base-config configs/multihead_v1.json \\
        --n-trials 20 \\
        --experiment pii-hparam-search \\
        --output-root hparam_outputs

Search space
------------
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
  max_span_len           categorical [8, 10, 12, 16]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import optuna

from multihead_pii.config import load_config

# Silence Optuna's per-trial INFO logs; keep WARNING+ only.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Try to use MLflow 3.x MlflowStorage for study persistence.
# Falls back to Optuna's default in-memory storage when not available.
# ---------------------------------------------------------------------------
try:
    from mlflow.optuna import MlflowStorage as _MlflowStorage
    _MLFLOW_STORAGE_AVAILABLE = True
except ImportError:
    _MLFLOW_STORAGE_AVAILABLE = False


def _suggest_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Map the search space onto Optuna's suggest_* API."""
    return {
        # Tier 1
        "learning_rate":           trial.suggest_float("learning_rate",        1e-5,  5e-5, log=True),
        "dropout":                 trial.suggest_float("dropout",               0.05,  0.30),
        "weight_decay":            trial.suggest_float("weight_decay",          1e-3,  0.10, log=True),
        # Tier 2
        "warmup_ratio":            trial.suggest_float("warmup_ratio",          0.05,  0.20),
        "negative_sample_rate":    trial.suggest_float("negative_sample_rate",  0.10,  0.50),
        # Tier 3
        "proposal_loss_weight":    trial.suggest_float("proposal_loss_weight",  0.5,   2.0),
        "type_loss_weight":        trial.suggest_float("type_loss_weight",      0.5,   2.0),
        "sensitivity_loss_weight": trial.suggest_float("sensitivity_loss_weight", 0.5, 2.0),
        # Tier 4
        "max_span_len":            trial.suggest_categorical("max_span_len",    [8, 10, 12, 16]),
    }


def _build_trial_config(base_config_path: str, hparams: Dict[str, Any]) -> Dict[str, Any]:
    cfg_dict = asdict(load_config(base_config_path))
    cfg_dict.update(hparams)
    return cfg_dict


def _read_best_valid_loss(trial_output: Path) -> float:
    history_path = trial_output / "train_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                return min(row["valid"]["loss"] for row in history)
        except Exception:
            pass
    return float("inf")


def _make_objective(args: argparse.Namespace, tmp_dir: Path):
    """Return an Optuna objective closure over the fixed CLI args."""

    def objective(trial: optuna.Trial) -> float:
        hparams = _suggest_hparams(trial)
        run_name = f"trial-{trial.number:03d}"
        trial_output = Path(args.output_root) / run_name
        trial_output.mkdir(parents=True, exist_ok=True)

        # Write per-trial config
        cfg_path = tmp_dir / f"{run_name}_config.json"
        cfg_path.write_text(
            json.dumps(_build_trial_config(args.base_config, hparams), indent=2),
            encoding="utf-8",
        )

        print(f"\n{'='*60}")
        print(f"[hparam_search] Optuna trial {trial.number}  ({run_name})")
        print(f"  hparams: {json.dumps(hparams, indent=4)}")
        print(f"{'='*60}\n")

        # ----------------------------------------------------------------
        # Open an MLflow run for this trial.
        # Training is launched with --no-mlflow so all logging lives here.
        # ----------------------------------------------------------------
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(hparams)
            mlflow.set_tag("optuna_trial_number", str(trial.number))

            cmd = [
                sys.executable, "-m", "multihead_pii.train",
                "--train",  args.train,
                "--valid",  args.valid,
                "--config", str(cfg_path),
                "--output", str(trial_output),
                "--no-mlflow",          # avoid double-logging; this run owns it
            ]
            if args.train_sensitivity:
                cmd += ["--train-sensitivity", args.train_sensitivity]
            if args.valid_sensitivity:
                cmd += ["--valid-sensitivity", args.valid_sensitivity]

            proc = subprocess.run(cmd, capture_output=False, text=True)

            best_valid_loss = _read_best_valid_loss(trial_output)

            # Log per-epoch metrics from train_history into the MLflow run
            history_path = trial_output / "train_history.json"
            if history_path.exists():
                try:
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    for row in history:
                        step = row["epoch"]
                        mlflow.log_metrics(
                            {
                                "train_loss":             row["train"]["loss"],
                                "train_proposal_loss":    row["train"]["proposal_loss"],
                                "train_type_loss":        row["train"]["type_loss"],
                                "train_sensitivity_loss": row["train"]["sensitivity_loss"],
                                "valid_loss":             row["valid"]["loss"],
                                "valid_proposal_loss":    row["valid"]["proposal_loss"],
                                "valid_type_loss":        row["valid"]["type_loss"],
                                "valid_sensitivity_loss": row["valid"]["sensitivity_loss"],
                            },
                            step=step,
                        )
                    mlflow.log_artifact(str(history_path), artifact_path="training")
                except Exception:
                    pass

            checkpoint = trial_output / "multihead_model.pt"
            if checkpoint.exists():
                mlflow.log_artifact(str(checkpoint), artifact_path="model")

            mlflow.log_metric("best_valid_loss", best_valid_loss)
            mlflow.log_metric("returncode", proc.returncode)
            mlflow.set_tag("mlflow_run_id", run.info.run_id)

        if proc.returncode != 0:
            raise optuna.exceptions.TrialPruned(
                f"Training subprocess exited with code {proc.returncode}"
            )

        return best_valid_loss

    return objective


def _create_study(args: argparse.Namespace) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    if _MLFLOW_STORAGE_AVAILABLE:
        # MLflow 3.x: persist the study to the tracking server so it can be
        # resumed across sessions and inspected in the MLflow UI.
        storage = _MlflowStorage(experiment_name=args.experiment)
        print(f"[hparam_search] Using MlflowStorage for study persistence.")
    else:
        storage = None

    study = optuna.create_study(
        study_name=args.experiment,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,    # resume if a study with this name exists
    )
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search (Optuna TPE) for MultiHeadPiiModel.",
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
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--experiment", default="pii-hparam-search",
        help="MLflow experiment name (and Optuna study name) shared by all trials.",
    )
    parser.add_argument(
        "--output-root", default="hparam_outputs",
        help="Root directory; one subdirectory is created per trial.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the Optuna TPE sampler.",
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


def print_summary(study: optuna.Study) -> None:
    trials = [t for t in study.trials if t.value is not None]
    if not trials:
        print("No completed trials.")
        return

    sorted_trials = sorted(trials, key=lambda t: t.value)
    col_w = 14
    header = (
        f"{'Trial':<12} {'best_val_loss':>{col_w}} "
        f"{'lr':>{col_w}} {'dropout':>{col_w}} {'wd':>{col_w}} "
        f"{'warmup':>{col_w}} {'neg_rate':>{col_w}} "
        f"{'prop_w':>{col_w}} {'type_w':>{col_w}} {'sens_w':>{col_w}} "
        f"{'span_len':>{col_w}}"
    )
    print("\n" + "=" * len(header))
    print("HYPERPARAMETER SEARCH SUMMARY  (sorted by best_val_loss, Optuna TPE)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for t in sorted_trials:
        p = t.params
        print(
            f"{'trial-' + str(t.number).zfill(3):<12} {t.value:>{col_w}.6f} "
            f"{p.get('learning_rate', float('nan')):>{col_w}.2e} "
            f"{p.get('dropout', float('nan')):>{col_w}.3f} "
            f"{p.get('weight_decay', float('nan')):>{col_w}.4f} "
            f"{p.get('warmup_ratio', float('nan')):>{col_w}.3f} "
            f"{p.get('negative_sample_rate', float('nan')):>{col_w}.3f} "
            f"{p.get('proposal_loss_weight', float('nan')):>{col_w}.3f} "
            f"{p.get('type_loss_weight', float('nan')):>{col_w}.3f} "
            f"{p.get('sensitivity_loss_weight', float('nan')):>{col_w}.3f} "
            f"{p.get('max_span_len', '?'):>{col_w}}"
        )
    print("=" * len(header))
    best = study.best_trial
    print(f"\nBest trial : trial-{str(best.number).zfill(3)}  (valid_loss={best.value:.6f})")
    print(f"Hparams    : {json.dumps(best.params, indent=4)}")


def main() -> None:
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    study = _create_study(args)
    n_existing = len([t for t in study.trials if t.value is not None])
    if n_existing:
        print(f"[hparam_search] Resuming study '{args.experiment}' "
              f"({n_existing} completed trials already).")

    with tempfile.TemporaryDirectory() as tmp_dir:
        study.optimize(
            _make_objective(args, Path(tmp_dir)),
            n_trials=args.n_trials,
            catch=(Exception,),     # log failed trials, don't abort the study
        )

    print_summary(study)

    # Dump results to JSON for offline analysis
    results = [
        {
            "trial": f"trial-{str(t.number).zfill(3)}",
            "best_valid_loss": t.value,
            "hparams": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]
    summary_path = Path(args.output_root) / "search_results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
