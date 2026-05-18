"""Unit tests for src/explain.py."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.explain import explain
from src.preprocessing import build_pipeline

_SPAM_DOCS = [
    "free prize win money claim now",
    "limited offer click here reward",
    "congratulations winner selected",
]
_HAM_DOCS = [
    "meeting agenda thursday project",
    "call scheduled review document",
    "quarterly report attached please",
]
_X = _SPAM_DOCS + _HAM_DOCS
_Y = [1, 1, 1, 0, 0, 0]


@pytest.fixture(scope="module")
def logreg_pipe():
    pipe = build_pipeline(LogisticRegression(random_state=42))
    pipe.set_params(tfidf__min_df=1)
    pipe.fit(_X, _Y)
    return pipe


@pytest.fixture(scope="module")
def nb_pipe():
    pipe = build_pipeline(MultinomialNB())
    pipe.set_params(tfidf__min_df=1)
    pipe.fit(_X, _Y)
    return pipe


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_list(self, logreg_pipe):
        result = explain(logreg_pipe, "free money click")
        assert isinstance(result, list)

    def test_each_item_is_tuple_of_str_and_float(self, logreg_pipe):
        result = explain(logreg_pipe, "free money click")
        for token, score in result:
            assert isinstance(token, str)
            assert isinstance(score, float)

    def test_respects_top_n(self, logreg_pipe):
        result = explain(logreg_pipe, "free prize win money claim", top_n=3)
        assert len(result) <= 3

    def test_default_top_n_is_10(self, logreg_pipe):
        result = explain(logreg_pipe, "free prize win money claim now")
        assert len(result) <= 10

    def test_fewer_tokens_than_top_n(self, logreg_pipe):
        result = explain(logreg_pipe, "free", top_n=10)
        assert len(result) <= 10

    def test_empty_text_returns_empty_list(self, logreg_pipe):
        result = explain(logreg_pipe, "")
        assert result == []


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_sorted_by_descending_absolute_score(self, logreg_pipe):
        result = explain(logreg_pipe, "free prize win money claim now report meeting")
        abs_scores = [abs(score) for _, score in result]
        assert abs_scores == sorted(abs_scores, reverse=True)

    def test_sorted_by_descending_absolute_score_nb(self, nb_pipe):
        result = explain(nb_pipe, "free prize win money claim now report meeting")
        abs_scores = [abs(score) for _, score in result]
        assert abs_scores == sorted(abs_scores, reverse=True)


# ---------------------------------------------------------------------------
# Score signs
# ---------------------------------------------------------------------------

class TestScoreSigns:
    def test_spam_tokens_have_positive_scores_logreg(self, logreg_pipe):
        result = explain(logreg_pipe, "free money click")
        scores = {token: score for token, score in result}
        # All recognised spam tokens should push toward spam (positive)
        for token in ("free", "money", "click"):
            if token in scores:
                assert scores[token] > 0, f"expected positive score for spam token '{token}'"

    def test_ham_tokens_have_negative_scores_logreg(self, logreg_pipe):
        result = explain(logreg_pipe, "meeting agenda report")
        scores = {token: score for token, score in result}
        for token in ("meet", "agenda", "report"):
            if token in scores:
                assert scores[token] < 0, f"expected negative score for ham token '{token}'"

    def test_spam_tokens_have_positive_scores_nb(self, nb_pipe):
        result = explain(nb_pipe, "free money click")
        scores = {token: score for token, score in result}
        for token in ("free", "money", "click"):
            if token in scores:
                assert scores[token] > 0

    def test_ham_tokens_have_negative_scores_nb(self, nb_pipe):
        result = explain(nb_pipe, "meeting agenda report")
        scores = {token: score for token, score in result}
        for token in ("meet", "agenda", "report"):
            if token in scores:
                assert scores[token] < 0


# ---------------------------------------------------------------------------
# Both classifiers produce results
# ---------------------------------------------------------------------------

class TestClassifierSupport:
    def test_works_with_logistic_regression(self, logreg_pipe):
        result = explain(logreg_pipe, "free prize click")
        assert len(result) > 0

    def test_works_with_multinomial_nb(self, nb_pipe):
        result = explain(nb_pipe, "free prize click")
        assert len(result) > 0

    def test_unsupported_classifier_raises_type_error(self):
        pipe = build_pipeline(LinearSVC())
        pipe.set_params(tfidf__min_df=1)
        pipe.fit(_X, _Y)
        with pytest.raises(TypeError, match="Unsupported classifier"):
            explain(pipe, "free money click")
