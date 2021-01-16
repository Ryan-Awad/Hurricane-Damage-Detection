import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

### Excluding Imports ###
st.title("Hurrican Damage Detection")
st.write("Hi! This model takes satellite images from earth and depicts whether they have been damaged by a satellite. To begin, upload a png or jpeg file below.")

image = st.file_uploader(label="Upload your satellite image", type=["png", "jpeg", "jpg"])

st.image(image)
