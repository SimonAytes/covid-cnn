import streamlit as st
import tensorflow as tf

import numpy as np
from PIL import Image  # Strreamlit works with PIL library very easily for Images
import cv2

model_path = './models/cnn-covid-77.h5'

st.title("Pulmonary CT-Scan Diagnostic Tool")
st.write("DISCLAIMER: This is not a medically validated tool and should be used for educational purposes only.")

st.subheader("Upload CT-Scan File")
upload = st.file_uploader('Upload a CT scan image')
if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # Color from BGR to RGB
    img = Image.open(upload)

    st.subheader("Selected Image")
    st.image(img, caption='Uploaded Image', width=300)

    st.subheader("Analyze")
    if st.button('Analyze CT-Scan'):
        model = tf.keras.models.load_model(model_path)
        x = cv2.resize(opencv_image, (100, 100))
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)
        ans = np.argmax(y, axis=1)

        if ans == 0:
            st.write(f'**Result:** COVID infection')
        elif ans == 1:
            st.write(f'**Result:** Healthy')
        else:
            st.write(f'**Result:** Other Pulmonary Disorder')
