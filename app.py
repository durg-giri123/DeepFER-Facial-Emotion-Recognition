"""
DeepFER - Facial Emotion Recognition
------------------------------------
Streamlit Application
"""

import streamlit as st

from utils.preprocess import preprocess_image
from utils.predict import predict_emotion
from utils.helpers import (
    get_emoji,
    format_percentage,
    format_time,
)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="wide"
)

# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.title("😊 DeepFER")

st.markdown(
    """
Facial Emotion Recognition using **Deep Learning (CNN)**

Upload a facial image and let the trained model predict the emotion.
"""
)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.header("📌 Model Information")

    st.write("**Model:** CNN")
    st.write("**Framework:** TensorFlow 2.20")
    st.write("**Input Size:** 48 × 48")
    st.write("**Color Mode:** Grayscale")
    st.write("**Classes:** 7 Emotions")

    st.divider()

    st.write("### Supported Emotions")

    emotions = [
        "😠 Angry",
        "🤢 Disgust",
        "😨 Fear",
        "😊 Happy",
        "😐 Neutral",
        "😢 Sad",
        "😲 Surprise"
    ]

    for emotion in emotions:
        st.write(emotion)

# ----------------------------------------------------
# Upload Image
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if uploaded_file is not None:

    processed_image, display_image = preprocess_image(uploaded_file)

    col1, col2 = st.columns([1, 1])

    # ---------------- Left Column ---------------- #

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            display_image,
            caption="Uploaded Image"
)

    # ---------------- Right Column ---------------- #

    with col2:

        st.subheader("Prediction")

        if st.button("🔍 Predict Emotion", use_container_width=True):

            result = predict_emotion(processed_image)

            emoji = get_emoji(result["emotion"])

            st.success(
                f"{emoji} {result['emotion']}"
            )

            st.metric(
                label="Confidence",
                value=format_percentage(result["confidence"])
            )

            st.divider()

            st.subheader("Top 3 Predictions")

            for pred in result["top_predictions"]:

                st.write(
                    f"{get_emoji(pred['emotion'])} "
                    f"{pred['emotion']} "
                    f"({format_percentage(pred['probability'])})"
                )

            st.divider()

            st.info(
                f"⚡ Inference Time: "
                f"{format_time(result['inference_time'])}"
            )