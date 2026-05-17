"""Evaluate saved spam classifier models and produce metric reports."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _ROOT / "models"
_DEFAULT_REPORTS_DIR = _ROOT / "reports"

_MODEL_KEYS = [
    ("logreg", "logreg.joblib", "Logistic Regression"),
    ("naive_bayes", "naive_bayes.joblib", "Multinomial NB"),
]


def evaluate(
    models_dir: Path = _DEFAULT_MODELS_DIR,
    reports_dir: Path = _DEFAULT_REPORTS_DIR,
) -> dict:
    """Load saved models and split data, compute metrics, save plots and JSON.

    Returns a dict keyed by model name with metric sub-dicts.
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_path = models_dir / "train_test_split.joblib"
    logger.info("Loading split data from %s", split_path)
    split = joblib.load(split_path)
    X_test = split["X_test"]
    y_test = split["y_test"]

    results: dict = {}
    pipelines: dict = {}

    for key, filename, label in _MODEL_KEYS:
        path = models_dir / filename
        logger.info("Loading %s from %s", label, path)
        pipe = joblib.load(path)
        pipelines[key] = (pipe, label)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        report = classification_report(
            y_test, y_pred, target_names=["ham", "spam"], output_dict=True
        )
        ap = average_precision_score(y_test, y_prob)

        print(f"\n{label} — test set results:")
        print(
            classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4)
        )
        print(f"  Average precision (PR AUC): {ap:.4f}")

        results[key] = {
            "model": label,
            "accuracy": report["accuracy"],
            "ham": {
                "precision": report["ham"]["precision"],
                "recall": report["ham"]["recall"],
                "f1": report["ham"]["f1-score"],
            },
            "spam": {
                "precision": report["spam"]["precision"],
                "recall": report["spam"]["recall"],
                "f1": report["spam"]["f1-score"],
            },
            "average_precision": ap,
        }

    _plot_confusion_matrices(pipelines, X_test, y_test, reports_dir)
    _plot_pr_curves(pipelines, X_test, y_test, reports_dir)
    _save_metrics_json(results, reports_dir)

    return results


def _plot_confusion_matrices(
    pipelines: dict,
    X_test,
    y_test,
    reports_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (key, filename, label) in zip(axes, _MODEL_KEYS):
        pipe, display_label = pipelines[key]
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["ham", "spam"],
            yticklabels=["ham", "spam"],
            ax=ax,
            cbar=False,
        )
        ax.set_title(display_label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    out = reports_dir / "eval_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def _plot_pr_curves(
    pipelines: dict,
    X_test,
    y_test,
    reports_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["steelblue", "darkorange"]
    for (key, filename, label), color in zip(_MODEL_KEYS, colors):
        pipe, display_label = pipelines[key]
        y_prob = pipe.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, color=color, lw=2, label=f"{display_label} (AP={ap:.4f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    out = reports_dir / "eval_pr_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def _save_metrics_json(results: dict, reports_dir: Path) -> None:
    out = reports_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved spam classifier models."
    )
    parser.add_argument(
        "--models-dir",
        default=str(_DEFAULT_MODELS_DIR),
        help=f"Directory containing .joblib files (default: {_DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(_DEFAULT_REPORTS_DIR),
        help=f"Directory to write plots and metrics.json (default: {_DEFAULT_REPORTS_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    evaluate(models_dir=Path(args.models_dir), reports_dir=Path(args.reports_dir))
