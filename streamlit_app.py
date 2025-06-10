import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(page_title="Digit Recognizer", layout="wide")

# Load your trained model
model = keras.models.load_model("model.h5")

# Header
st.markdown("<h1 style='text-align: center;'>Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Draw a digit (0–9) and click <b>Predict</b> to classify it.</p>", unsafe_allow_html=True)

# Side-by-side layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Processed Preview and Prediction")
    if st.button("Predict"):
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 0] < 255):
            # Convert canvas image to grayscale and preprocess
            img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8')).resize((28, 28)).convert('L')
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.image(img.resize((64, 64)), caption="Processed Image", width=64)
            st.markdown(f"<h3 style='color:green;'>Predicted Digit: {predicted_digit}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
        else:
            st.error("Please draw a digit on the canvas before clicking Predict.")
