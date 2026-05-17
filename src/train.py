"""Train LogisticRegression and MultinomialNB spam classifiers."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from src.data_loader import load_enron_spam
from src.preprocessing import build_pipeline

logger = logging.getLogger(__name__)

_ALL_SUBSETS = [f"enron{i}" for i in range(1, 7)]
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _ROOT / "models"


def _split(df: pd.DataFrame, test_subset: str) -> tuple[
    pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
]:
    """Split df into train/test by subset name.

    Returns (X_train, X_test, y_train, y_test, source_test).
    """
    train_mask = df["source"] != test_subset
    test_mask = ~train_mask

    X_train = df.loc[train_mask, "text"].reset_index(drop=True)
    X_test  = df.loc[test_mask,  "text"].reset_index(drop=True)
    y_train = df.loc[train_mask, "label"].reset_index(drop=True)
    y_test  = df.loc[test_mask,  "label"].reset_index(drop=True)
    source_test = df.loc[test_mask, "source"].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, source_test


def _train_and_report(
    name: str,
    pipeline,
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> object:
    """Fit pipeline, print a brief classification report, return fitted pipeline."""
    logger.info("Training %s ...", name)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("  done in %.1f s", elapsed)

    y_pred = pipeline.predict(X_test)
    print(f"\n{name} — test set results:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))
    return pipeline


def train(
    test_subset: str = "enron6",
    output_dir: Path = _DEFAULT_MODELS_DIR,
) -> dict[str, object]:
    """Load data, train both models, save artifacts, return fitted pipelines.

    Parameters
    ----------
    test_subset : one of enron1–enron6; held out entirely for evaluation.
    output_dir  : directory to write .joblib files to.

    Returns
    -------
    dict with keys 'logreg', 'naive_bayes', and 'split'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset ...")
    df = load_enron_spam()

    logger.info("Splitting: train on all except '%s', test on '%s'", test_subset, test_subset)
    X_train, X_test, y_train, y_test, source_test = _split(df, test_subset)

    train_subsets = [s for s in _ALL_SUBSETS if s != test_subset]
    logger.info(
        "  Train: %d emails from %s",
        len(X_train), train_subsets,
    )
    logger.info(
        "  Test : %d emails from %s  (ham=%d, spam=%d)",
        len(X_test), test_subset,
        (y_test == 0).sum(), (y_test == 1).sum(),
    )

    # ── Logistic Regression ───────────────────────────────────────────────
    logreg_pipeline = build_pipeline(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    )
    logreg_pipeline = _train_and_report(
        "LogisticRegression", logreg_pipeline, X_train, X_test, y_train, y_test
    )

    # ── Multinomial Naive Bayes ───────────────────────────────────────────
    nb_pipeline = build_pipeline(MultinomialNB(alpha=1.0))
    nb_pipeline = _train_and_report(
        "MultinomialNB", nb_pipeline, X_train, X_test, y_train, y_test
    )

    # ── Save artifacts ────────────────────────────────────────────────────
    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "source_test": source_test,
        "test_subset": test_subset,
        "train_subsets": train_subsets,
    }

    logreg_path = output_dir / "logreg.joblib"
    nb_path     = output_dir / "naive_bayes.joblib"
    split_path  = output_dir / "train_test_split.joblib"

    joblib.dump(logreg_pipeline, logreg_path)
    logger.info("Saved %s", logreg_path)

    joblib.dump(nb_pipeline, nb_path)
    logger.info("Saved %s", nb_path)

    joblib.dump(split_data, split_path)
    logger.info("Saved %s", split_path)

    return {"logreg": logreg_pipeline, "naive_bayes": nb_pipeline, "split": split_data}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LogReg and Naive Bayes spam classifiers on Enron-Spam."
    )
    parser.add_argument(
        "--test-subset",
        default="enron6",
        choices=_ALL_SUBSETS,
        help="Enron subset held out for testing (default: enron6).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_MODELS_DIR),
        help=f"Directory to write model artifacts to (default: {_DEFAULT_MODELS_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    train(test_subset=args.test_subset, output_dir=Path(args.output_dir))
