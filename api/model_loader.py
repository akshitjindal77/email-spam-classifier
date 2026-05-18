"""Load trained classifier pipelines from disk."""

from __future__ import annotations

from pathlib import Path

import joblib

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_models() -> dict[str, object]:
    """Load and return both fitted pipelines keyed by model name."""
    return {
        "logreg": joblib.load(_MODELS_DIR / "logreg.joblib"),
        "naive_bayes": joblib.load(_MODELS_DIR / "naive_bayes.joblib"),
    }
