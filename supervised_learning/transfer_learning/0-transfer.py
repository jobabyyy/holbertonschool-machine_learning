#!/usr/bin/env python3
"""Task 0: Transfer Learning - Training Model using
EfficientNetB1 Application"""


import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model


# Function to preprocess the data
def preprocess_data(X, Y):
    # Normalize pixel values to [0, 1]
    X_normalized = X.astype('float32') / 255.0

    # Return preprocessed data
    return X_normalized, Y


# Load CIFAR-10 data
(X_train, Y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
X_train, Y_train = preprocess_data(X_train, Y_train)

# Lambda layer to scale up the data to the correct size
resize_lambda = Lambda(lambda image: tf.image.resize(image,
                       (128, 128)))(X_train)

# Loading the pre-trained EfficientNetB1 model
base_model = EfficientNetB1(include_top=False,
                            weights='imagenet',
                            input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(10, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the custom classification layers on top of the base model
model.fit(resize_lambda, Y_train, batch_size=128,
          epochs=10, validation_split=0.2)

# Save the trained model
model.save('cifar10_model.h5')
print("Trained model saved.")
