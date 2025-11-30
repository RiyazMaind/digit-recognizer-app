# Digit Recognition App

This project is an interactive Digit Recognition Web Application built using Streamlit.
Users can draw a digit (0–9) on a digital canvas, and a trained TensorFlow/Keras CNN model predicts the number.

---

## Features

- Interactive canvas for drawing digits
- Real-time preprocessing (grayscale, resizing, normalization)
- TensorFlow CNN model prediction
- Lightweight and easy to deploy
- Suitable for Streamlit Cloud or Hugging Face Spaces

---

## Model Information

The model is a Convolutional Neural Network trained on the MNIST dataset (28×28 grayscale digit images).

Typical architecture used:

- Conv2D → ReLU
- MaxPooling2D
- Conv2D → ReLU
- MaxPooling2D
- Flatten
- Dense → Dropout
- Dense (Softmax output)

Model is saved as:

```
cnn_model.h5
```

---

## Tech Stack

- Streamlit
- streamlit-drawable-canvas
- TensorFlow / Keras
- NumPy
- Pillow (PIL)

---

## Project Structure

```
.
├── app.py
├── cnn_model.h5
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/digit-recognition-app.git
cd digit-recognition-app
```

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## How It Works

1. The user draws a digit on the canvas.
2. The drawing is converted to a 28×28 grayscale array.
3. The array is normalized and reshaped for the model.
4. The CNN model predicts the digit.
5. The prediction is displayed on the screen.

---

## Deployment

This application can be deployed easily using:

- Streamlit Cloud
- Hugging Face Spaces
- Any server or local machine
