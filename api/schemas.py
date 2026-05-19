"""Pydantic request and response schemas for the spam classifier API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw email text to classify.")
    model: Literal["naive_bayes", "logreg"] = Field(
        "naive_bayes",
        description="Classifier to use. naive_bayes (default) or logreg.",
    )


class TokenScore(BaseModel):
    token: str
    score: float


class PredictResponse(BaseModel):
    label: Literal["ham", "spam"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: Literal["naive_bayes", "logreg"]
    top_tokens: list[TokenScore]
