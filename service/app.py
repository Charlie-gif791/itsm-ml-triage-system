import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from model.load import predict, load_inference_bundle
from policy.decision import apply_decision
from config import PROJECT_ROOT, DEVICE, LABEL_MAP_PATH, THRESHOLD

app = FastAPI(title="ITSM ML Triage API")

bundle = load_inference_bundle(
    label_map_path=PROJECT_ROOT / LABEL_MAP_PATH,
    device=DEVICE,
)

class TicketRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    abstained: bool

@app.post("/predict", response_model=PredictionResponse)
def classify_ticket(request: TicketRequest):
    raw = predict(
        request.text,
        bundle=bundle,
    )

    final = apply_decision(
        raw["predicted_label"],
        raw["confidence"],
        threshold=THRESHOLD,
    )

    return final
