import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.layers.experimental.preprocessing as augment
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_img(img, label):
    img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 255.0 # convert image dtype to float32 and scaling values between 0 and 1
    return img, label

# Creating pipelines
train_pipeline = image_dataset_from_directory(
    'data/train_another',
    label_mode='binary',
    shuffle=True,
    batch_size=64,
    image_size=(128,128),
    color_mode='rgb'
).map(preprocess_img)

val_pipeline = image_dataset_from_directory(
    'data/validation_another',
    label_mode='binary',
    shuffle=False,
    batch_size=64,
    image_size=(128,128),
    color_mode='rgb'
).map(preprocess_img)

# Defining a model. We chose a convolutional neural network
model = keras.Sequential([ 
    layers.Input((128,128,3)),

    # Making the base
    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)), 
    layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)), 
    layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)), 
    
    # Making the head
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid') # used sigmoid as activation function because we are using binary labels
])

OPTIMIZER = keras.optimizers.Adam(
    learning_rate=1e-4
)
LOSS = keras.losses.binary_crossentropy
model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=['binary_accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    min_delta=1e-3,
    patience=4,
    verbose=1,
    restore_best_weights=True
)

ckpt = ModelCheckpoint(
    'model_weights/hurricane-model-weights.hdf5',
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

history = model.fit(
    train_pipeline,
    validation_data=(val_pipeline),
    batch_size=64,
    callbacks=[early_stopping, ckpt],
    verbose=1,
    epochs=200
)

fit_hist = pd.DataFrame(history.history)
fit_hist.to_csv('model_data/fit_history.csv')

loss = round(np.min(fit_hist['loss']), 2)
val_loss = round(np.min(fit_hist['val_loss']), 2)
acc = round(np.max(fit_hist['binary_accuracy']), 2)
val_acc = round(np.max(fit_hist['val_binary_accuracy']), 2)

plt.title(f"Train Loss ({loss}) and Validation Loss ({val_loss})")
plt.plot(fit_hist['loss'], label='Train Loss')
plt.plot(fit_hist['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(color='#e6e6e6')
plt.legend()
plt.show()

plt.title(f"Train Accuracy ({acc}) and Validation Accuracy ({val_acc})")
plt.plot(fit_hist['binary_accuracy'], label='Train Acc')
plt.plot(fit_hist['val_binary_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(color='#e6e6e6')
plt.legend()
plt.show()
