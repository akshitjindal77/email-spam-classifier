"""FastAPI spam classifier application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.model_loader import load_models
from api.schemas import PredictRequest, PredictResponse, TokenScore
from src.explain import explain

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
    yield


app = FastAPI(
    title="Spam Classifier API",
    description="Classifies email text as ham or spam using TF-IDF + ML models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_FRONTEND / "index.html")


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    pipeline = request.app.state.models[body.model]
    proba = pipeline.predict_proba([body.text])[0]
    predicted_class = int(proba.argmax())
    label = "spam" if predicted_class == 1 else "ham"
    confidence = round(float(proba[predicted_class]), 4)
    top_tokens = [TokenScore(token=t, score=round(s, 4)) for t, s in explain(pipeline, body.text)]
    return PredictResponse(label=label, confidence=confidence, model=body.model, top_tokens=top_tokens)
