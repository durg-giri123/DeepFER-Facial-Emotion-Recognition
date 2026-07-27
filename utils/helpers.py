"""
Helper utilities for the DeepFER Streamlit application.
"""

# ---------------------------------------------------
# Emotion → Emoji Mapping
# ---------------------------------------------------

EMOTION_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}


# ---------------------------------------------------
# Get Emoji
# ---------------------------------------------------

def get_emoji(emotion):
    """
    Return emoji corresponding to an emotion.
    """
    return EMOTION_EMOJIS.get(emotion, "🙂")


# ---------------------------------------------------
# Confidence Color
# ---------------------------------------------------

def confidence_color(confidence):
    """
    Return a color based on prediction confidence.

    Parameters
    ----------
    confidence : float
        Confidence percentage (0–100)

    Returns
    -------
    str
        Hex color code.
    """

    if confidence >= 90:
        return "#2E8B57"      # Green

    elif confidence >= 75:
        return "#FF9800"      # Orange

    else:
        return "#E53935"      # Red


# ---------------------------------------------------
# Format Inference Time
# ---------------------------------------------------

def format_time(seconds):
    """
    Convert seconds to a readable string.
    """

    return f"{seconds:.4f} sec"


# ---------------------------------------------------
# Format Percentage
# ---------------------------------------------------

def format_percentage(value):
    """
    Format percentage with two decimal places.
    """

    return f"{value:.2f}%"