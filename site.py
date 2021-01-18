import sys
print("SYS VERSION:::::::::::::::::::::::::::" + str(sys.version))

import os
print("OS FILE:::::::::::::::::::::::::::::::" + str(os.__file__))
 
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Hurricane Damage Detection",  # default page title
    layout="centered"
)

activity = st.sidebar.selectbox("Analyze", ["Model", "About", "Visualize the Data"])

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
    This model uses 3 channel (RGB) satellite images from Earth and detects whether the area in the image has been damaged by a Hurricane. To begin, upload a 3 channel (RGB) satellite image.   

    Check out the ***About*** and ***Visualize the Data*** pages in the sidebar for more information.
    """)

    image_file = st.file_uploader(label="Upload Satellite Image", type=["png", "jpeg", "jpg"])
    if image_file != None:
        with st.spinner(text='Making prediction...'):
            class_names = ("DAMAGE", "No damage")
            pred = predict(image_file)
            pred = class_names[int(round(pred[0][0]))]
            if pred == class_names[0]:
                st.success(f'**\* {pred} was detected in the uploaded image! * **')
            elif pred == class_names[1]:
                st.success(f'{pred} was detected in the uploaded image.')

    st.markdown('''
    [GitHub Repository](https://github.com/Ryan-Awad/Hurricane-Damage-Detection/tree/master) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Devpost Submission](https://devpost.com/software/hurricane-damage-detection-rt1sz6)

    [Personal GitHub](https://github.com/Ryan-Awad) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Linkedin](https://www.linkedin.com/in/ryanawad) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Personal Devpost](https://devpost.com/Ryan-Awad)
    ''')

elif activity == "About":
    st.title("About")
    st.write("""
    ### How the Model Works
    Hurricanes cause lots of devastation and affect many people physically, mentally and economically. For our model, we utilized a dataset that contains satellite images of both damaged and undamaged areas in Texas after Hurricane Harvey unfortunately struck in 2017. For this reason we decided that we would create an application that would detect whether an area has been damaged by a hurricane. This would be useful for detecting if an area was damaged by a hurricane much earlier. This is great because it allows help to be sent earlier to the designated damaged area. Instead of having to wait for a damaged area to be reported, which takes longer, you could get a live satellite image of each area and see exactly which areas need help. 
    
    ### More in Depth
    To build our model, we used *TensorFlow* to create a *convolutional neural network (CNN)* to detect patterns and edges in the satellite images. The optimizer used to compile model was the *Adam optimizer* with a learning rate of `1e-4`. For the loss function, we used *binary cross-entropy*. This allowed our model to achieve an accuracy of `96%`. For more info on the architecture of our model, check out our [GitHub repository](https://github.com/Ryan-Awad/Hurricane-Damage-Detection/tree/model-developpement). 

    ### Where We Got The Data
    The data is composed of satellite images from Texas after Hurricane Harvey struck in 2017. The data is divided into 2 groups, `damage` and `no_damage`. This data allows us to train a model to detect if an area in **any** satellite image was damaged by a hurricane. The data used to train the model was taken from [here](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized).
    """)

elif activity == "Visualize the Data":
    st.title("Visualize the Data")

    fit_data = pd.read_csv("visual_data/fit_history.csv")
    st.write('### Model Train and Validation Loss')
    st.line_chart(fit_data[["Train Loss", "Validation Loss"]])
    st.write('### Model Train and Validation Accuracy')
    st.line_chart(fit_data[["Train Accuracy", "Validation Accuracy"]])

    st.write('### Spatial Distribution of Dataset Images')
    @st.cache
    def load_data(path):
        map_data = pd.read_csv(path)
        return map_data

    map_df = load_data('visual_data/image_locations.csv')
    fig, ax = plt.subplots()
    sns.scatterplot(x=map_df['lon'], y=map_df['lat'], ax=ax, size=0.5, legend=False)
    plt.grid(color='#dddddd')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    st.pyplot(fig)
else:
    st.title("Page Not Found")


