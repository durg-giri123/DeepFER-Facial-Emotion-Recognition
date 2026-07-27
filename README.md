# 😊 DeepFER - Facial Emotion Recognition using Deep Learning

A real-time Facial Emotion Recognition web application built using **TensorFlow**, **CNN**, and **Streamlit**. The application predicts human emotions from facial images across seven emotion categories.

---

## 🚀 Live Demo

> Coming Soon (Hugging Face Spaces)

---

## 📌 Features

- Upload facial images
- Predict one of 7 emotions
- Display prediction confidence
- Show Top-3 predicted emotions
- Fast inference using a trained CNN model
- Clean Streamlit interface

---

## 🧠 Supported Emotions

- 😊 Happy
- 😐 Neutral
- 😡 Angry
- 😢 Sad
- 😨 Fear
- 🤢 Disgust
- 😲 Surprise

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pillow

---

## 📂 Project Structure

```
DeepFER/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
├── utils/
├── dataset/
├── notebooks/
├── Outputs/
├── Report/
└── Screenshots/
```

---

## 📂 Dataset

This project was trained on the **FER2013** dataset.

**Dataset Link:**
https://www.kaggle.com/datasets/msambare/fer2013

To retrain the model:

1. Download the dataset from the above link.
2. Extract it into the `dataset/` folder.
3. Run the training notebook (`FacialEmotionDetection.ipynb`).


## ⚙️ Installation

Clone the repository

```bash
git clone <your-repository-url>
cd DeepFER
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Model Information

- Framework: TensorFlow 2.20
- Model: CNN
- Input Size: 48 × 48
- Color Mode: Grayscale
- Classes: 7 Emotions

---

## 📈 Results

The application predicts:

- Primary emotion
- Confidence score
- Top 3 predictions
- Inference time

---

## 📸 Screenshots

(Add screenshots after deployment.)

---

## 👨‍💻 Author

**Durgesh Giri**

B.Tech CSE | AI & ML Enthusiast

GitHub: https://github.com/durg-giri123

LinkedIn: (Add your profile link)

---

## ⭐ Future Improvements

- Webcam-based emotion detection
- Face detection before classification
- Batch image prediction
- Model comparison dashboard
- Performance analytics