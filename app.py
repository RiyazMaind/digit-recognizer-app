import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import numpy as np
from PIL import Image

# Load trained model
model = keras.models.load_model("model.h5")

st.title("Draw a Digit")
st.markdown("Draw a digit (0–9) below and click **Predict** to classify it.")

# Create canvas component
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

if canvas_result.image_data is not None:
    # Convert drawing to grayscale image
    img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
    img = img.resize((28, 28))
    img = img.convert('L')

    # Normalize and reshape
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Show preview
    st.image(img, caption="Resized 28x28 Input", width=150)

    # Predict on button click
    if st.button("Predict"):
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        st.success(f"Predicted Digit: **{predicted_label}**")
