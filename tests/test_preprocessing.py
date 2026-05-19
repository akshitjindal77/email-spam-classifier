"""Unit tests for src/preprocessing.py."""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocessing import build_pipeline, clean_text


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_url_replaced(self):
        result = clean_text("free money at http://spam.com click now")
        assert "__url__" in result
        assert "http" not in result
        assert "spam.com" not in result
        assert "free" in result  # word outside the URL is preserved

    def test_www_url_replaced(self):
        result = clean_text("go to www.example.com today")
        assert "__url__" in result

    def test_email_replaced(self):
        result = clean_text("contact winner@spam.com for details")
        assert "__email__" in result
        assert "@" not in result

    def test_email_replaced_before_url(self):
        # Emails containing http-like strings shouldn't be double-replaced
        result = clean_text("reply to user@domain.com or visit http://site.com")
        assert "__email__" in result
        assert "__url__" in result
        assert "@" not in result

    def test_punctuation_stripped(self):
        result = clean_text("Hello, world! Buy now.")
        assert "," not in result
        assert "!" not in result
        assert "." not in result

    def test_digits_stripped(self):
        result = clean_text("Win $1000 in 24 hours")
        assert "1000" not in result
        assert "24" not in result

    def test_stemming_applied(self):
        result = clean_text("running quickly through beautiful gardens")
        assert "run" in result
        assert "garden" in result
        # Unstemmed forms should not appear
        assert "running" not in result
        assert "gardens" not in result

    def test_single_char_tokens_dropped(self):
        # Single letters should be filtered out
        result = clean_text("a b c hello world")
        tokens = result.split()
        assert all(len(t) > 1 for t in tokens if not t.startswith("__"))

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_punctuation(self):
        assert clean_text("!!! ??? ...") == ""

    def test_only_numbers(self):
        assert clean_text("123 456 789") == ""

    def test_placeholder_tokens_preserved(self):
        # __email__ and __url__ survive as single tokens
        result = clean_text("email me at user@host.com")
        assert "__email__" in result.split()

    def test_returns_string(self):
        assert isinstance(clean_text("hello world"), str)

    def test_idempotent(self):
        # Running clean_text twice should give the same result as once
        once = clean_text("Free money! Visit http://spam.com")
        twice = clean_text(once)
        assert once == twice


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_pipeline(self):
        pipe = build_pipeline(LogisticRegression())
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_tfidf_step(self):
        pipe = build_pipeline(LogisticRegression())
        assert "tfidf" in pipe.named_steps

    def test_pipeline_has_clf_step(self):
        pipe = build_pipeline(LogisticRegression())
        assert "clf" in pipe.named_steps

    def test_accepts_logistic_regression(self):
        pipe = build_pipeline(LogisticRegression())
        assert isinstance(pipe.named_steps["clf"], LogisticRegression)

    def test_accepts_naive_bayes(self):
        pipe = build_pipeline(MultinomialNB())
        assert isinstance(pipe.named_steps["clf"], MultinomialNB)

    def test_pipeline_fits_and_predicts(self):
        pipe = build_pipeline(LogisticRegression())
        pipe.set_params(tfidf__min_df=1)
        X = [
            "free prize win money claim now",
            "meeting agenda thursday project",
            "limited offer click here reward",
            "call scheduled review document",
            "congratulations winner selected",
            "quarterly report attached please",
        ]
        y = [1, 0, 1, 0, 1, 0]
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_pipeline_predict_proba(self):
        pipe = build_pipeline(LogisticRegression())
        pipe.set_params(tfidf__min_df=1)
        X = [
            "free money win prize click",
            "project update meeting notes",
            "claim reward limited offer",
            "report attached review please",
        ]
        y = [1, 0, 1, 0]
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (4, 2)
        # Each row must sum to ~1
        import numpy as np
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_tfidf_uses_clean_text_preprocessor(self):
        pipe = build_pipeline(LogisticRegression())
        preprocessor = pipe.named_steps["tfidf"].preprocessor
        assert preprocessor is clean_text

    def test_tfidf_sublinear_tf_enabled(self):
        pipe = build_pipeline(LogisticRegression())
        assert pipe.named_steps["tfidf"].sublinear_tf is True

    def test_tfidf_bigrams_enabled(self):
        pipe = build_pipeline(LogisticRegression())
        assert pipe.named_steps["tfidf"].ngram_range == (1, 2)

    def test_independent_pipelines_do_not_share_state(self):
        pipe1 = build_pipeline(LogisticRegression())
        pipe2 = build_pipeline(LogisticRegression())
        assert pipe1 is not pipe2
        assert pipe1.named_steps["tfidf"] is not pipe2.named_steps["tfidf"]
