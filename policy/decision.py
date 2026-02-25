def apply_decision(predicted_label, confidence, threshold):
    if confidence < threshold:
        return {
            "predicted_label": "ABSTAIN",
            "confidence": confidence,
            "abstained": True
        }
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "abstained": False
    }
