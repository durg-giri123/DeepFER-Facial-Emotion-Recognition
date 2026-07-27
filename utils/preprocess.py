"""
Image preprocessing utilities for DeepFER.

This module prepares uploaded images before they are passed
to the trained CNN model.
"""

import cv2
import numpy as np
from PIL import Image


IMAGE_SIZE = (48, 48)


def preprocess_image(uploaded_file):
    """
    Preprocess an uploaded image for emotion prediction.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Image uploaded through Streamlit.

    Returns
    -------
    processed_image : np.ndarray
        Image ready for CNN prediction.
        Shape -> (1, 48, 48, 1)

    display_image : PIL.Image
        Original image for displaying in the UI.
    """

    # Read uploaded image
    display_image = Image.open(uploaded_file).convert("RGB")

    # PIL → NumPy
    image = np.array(display_image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    gray = cv2.resize(gray, IMAGE_SIZE)

    # Normalize
    gray = gray.astype(np.float32) / 255.0

    # Add channel dimension
    gray = np.expand_dims(gray, axis=-1)

    # Add batch dimension
    processed_image = np.expand_dims(gray, axis=0)

    return processed_image, display_image