import torch
from pathlib import Path
from model.load import load_inference_bundle, predict
from config import DEVICE, PROJECT_ROOT, LABEL_MAP_PATH

if __name__ == "__main__":
    bundle = load_inference_bundle(
        label_map_path=PROJECT_ROOT / LABEL_MAP_PATH,
        device=DEVICE,
    )

    texts = [
        "Deployment failed after CI pipeline update",
        "User cannot access VPN",
    ]

    preds = predict(
        text=texts,
        bundle=bundle,
    )
    
    print(f"predicted_label: ",preds["predicted_label"])
    print(f"confidence: ",preds["confidence"])
    print(f"abstained: ", preds["abstained"])
