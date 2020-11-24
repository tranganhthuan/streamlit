import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 as cv

# Predict img with cache
@st.cache
def predict(img):
    img = img.astype('uint8')
    img = cv.resize(img,(28,28), interpolation = cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    predict = np.argmax(model.predict(img), axis=1)[0]
    return predict

# Load model
model = keras.models.load_model('lenet5_mnist')

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=25,
    stroke_color="#FFFFFF",
    background_color="#000000",
    update_streamlit=False,
    height=250,
    width=250,
    drawing_mode="freedraw",
    key="canvas",
)

# Create label
placeholder = st.empty()
placeholder.write('Number:')

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    img = canvas_result.image_data.copy()
    predict = predict(img)
    placeholder.write('Number: '+ str(predict))


