
import streamlit as st
import tensorflow as tf
from matplotlib import image
import numpy as np
from keras.models import load_model

st.header('Nepali Handwritten Digit Classifier')


upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
    img = image.imread(upload)
    img_batch = np.expand_dims(img,0)
    # prediction = MODEL.predict(img_batch)
    # predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    # confidence = np.max(prediction[0])

    MODEL = load_model("model.h5")
    prediction = MODEL.predict(img_batch)
    predicted_class = np.argmax(prediction[0])

    c1.image(upload)
    c1.header('Output')
    c1.subheader(f"Predicted digit : {predicted_class}")
    