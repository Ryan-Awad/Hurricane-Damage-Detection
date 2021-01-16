import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


st.title("Hurricane Damage Detection")
st.write("Hi! This model takes satellite images from earth and depicts whether they have been damaged by a Hurricane. To begin, upload a png or jpeg file below.")

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
    st.write("here is an explanation")
