import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Hurricane Damage Detection",  # default page title
    layout="centered"
)

activity = st.sidebar.selectbox("Analyze", ["Model", "Map", "Model Fitting Data"])

if activity == "Model":
    def process_img(img):
        img = Image.open(img)
        img = np.resize(img, (128,128,3))
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if float(np.max(img)) >= 255.0:
            img = img / 255
        img = tf.expand_dims(img, axis=0)
        return img

    def build_model(opt, loss):
        model = keras.Sequential([ 
            layers.Input((128,128,3)),
            layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=(2,2)), 
            layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=(2,2)), 
            layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=(2,2)), 
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid') # used sigmoid as activation function because we are using binary labels
        ])
        model.compile(optimizer=opt, loss=loss)
        return model

    @st.cache(allow_output_mutation=True)
    def model_cache():
        model = build_model(
            opt=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.binary_crossentropy
        )
        model.load_weights('model/hurricane-model-weights.hdf5')
        return model

    def predict(x):
        x = process_img(x)
        model = model_cache()
        pred = model.predict(x)
        return pred



    st.title("Hurricane Damage Detection")
    st.write("""
    This model takes satellite images from earth and depicts whether they have been damaged by a Hurricane. To begin, upload an image file below.   
    """)

    image_file = st.file_uploader(label="Upload your satellite image", type=["png", "jpeg", "jpg"])
    if image_file != None:
        with st.spinner(text='Making prediction...'):
            class_names = ("DAMAGE", "No damage")
            pred = predict(image_file)
            pred = class_names[int(round(pred[0][0]))]
            if pred == class_names[0]:
                st.success(f'**\* {pred} was detected in the uploaded image! * **')
            elif pred == class_names[1]:
                st.success(f'{pred} was detected in the uploaded image.')

elif activity == "Map":
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon']
    )
    st.map(map_data)

elif activity == "Model Fitting Data":
    st.write("in progress!")

else:
    st.title("Error 404: Page Not Found")
    st.write("I think you're a little lost...")


