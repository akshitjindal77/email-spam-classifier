"""Token-level explainability for TF-IDF spam classifier pipelines."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def explain(
    pipeline: Pipeline,
    text: str,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top_n most influential tokens for a single email.

    Scores are computed as tfidf_value × feature_weight, where feature_weight is:
      - LogisticRegression : coef_[0] (weight for spam class)
      - MultinomialNB      : log P(token|spam) − log P(token|ham)

    Returns
    -------
    List of (token, score) tuples sorted by descending |score|.
    Positive score → spam signal; negative score → ham signal.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    if isinstance(clf, LogisticRegression):
        weights = clf.coef_[0]
    elif isinstance(clf, MultinomialNB):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        raise TypeError(f"Unsupported classifier type: {type(clf).__name__}")

    feature_names = tfidf.get_feature_names_out()
    X = tfidf.transform([text]).tocoo()

    contributions = [
        (feature_names[col], float(val * weights[col]))
        for col, val in zip(X.col, X.data)
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]
