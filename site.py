import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

activity = st.sidebar.selectbox("Analyze", ["Model", "Map", "Model Fitting Data"])

if activity == "Model":
    st.title("Hurricane Damage Detection")
    st.write("""
    Hi! This model takes satellite images from earth and depicts whether they have been damaged by a Hurricane. To begin, upload a png or jpeg file below.   
    """)

    def load_image(image_file):
        img = Image.open(image_file)
        return img


    image_file = st.file_uploader(label="Upload your satellite image", type=["png", "jpeg", "jpg"])

    if image_file != None:
            st.image(load_image(image_file))

    col_1, col_2 = st.beta_columns([3,1])

    explanation = False

    with col_1:
        if st.button("view explanation"):
            explanation = True
            with col_2:
                if st.button("hide explanation"):
                    explanation = False

    if explanation:
        st.write("""
            Here is an explanation of what the model does. We have trained a neural network on a large
             dataset containing satellite imagery of areas that were damaged or damaged by a hurricane.
            After reaching an iteration with a high accuracy, the model stops training and the exact values 
            for the parameters(weights and biases) are saved so that we can consistently get a high accuracy 
            without having to retrain the model.
        """)

elif activity == "Map":
    map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
    st.map(map_data)

elif activity == "Model Fitting Data":
    st.write("in progress!")

else:
    st.title("Error 404: Page Not Found")
    st.write("I think you're a little lost...")


