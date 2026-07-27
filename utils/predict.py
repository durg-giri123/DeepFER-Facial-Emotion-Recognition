"""
Prediction utilities for DeepFER.

This module loads the trained CNN model and performs
emotion prediction on preprocessed images.
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf


# -----------------------------
# Emotion Labels
# -----------------------------
CLASS_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# -----------------------------
# Model Path
# -----------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "deepfer_best_model.keras"


# -----------------------------
# Load Model
# -----------------------------
_model = None


def load_model():
    """
    Load the trained CNN model.
    The model is loaded only once.
    """

    global _model

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model


# -----------------------------
# Predict Emotion
# -----------------------------
def predict_emotion(processed_image):
    """
    Predict emotion from a preprocessed image.

    Parameters
    ----------
    processed_image : np.ndarray
        Shape -> (1, 48, 48, 1)

    Returns
    -------
    dict
        Dictionary containing prediction results.
    """

    model = load_model()

    start_time = time.perf_counter()

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    inference_time = time.perf_counter() - start_time

    predicted_index = np.argmax(probabilities)

    predicted_emotion = CLASS_LABELS[predicted_index]

    confidence = float(probabilities[predicted_index] * 100)

    # Sort probabilities
    sorted_indices = np.argsort(probabilities)[::-1]

    top_predictions = []

    for idx in sorted_indices[:3]:
        top_predictions.append({
            "emotion": CLASS_LABELS[idx],
            "probability": float(probabilities[idx] * 100)
        })

    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
        "inference_time": inference_time
    }