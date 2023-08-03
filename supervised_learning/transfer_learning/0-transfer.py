#!/usr/bin/env python3
"""Task 0: Transfer Learning - Training Model using
EfficientNetB0 Application"""


import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Function to preprocess the data
def preprocess_data(X, Y):
    # Normalize pixel values to [0, 1]
    X_normalized = X.astype('float32') / 255.0

    # Return preprocessed data
    return X_normalized, Y


# Load CIFAR-10 data
(X_train, Y_train), (_, _) = cifar10.load_data()

# Preprocess the data
X_train, Y_train = preprocess_data(X_train, Y_train)

# Resize images to (224, 224) for EfficientNetB0
X_train_resized = tf.image.resize(X_train, (224, 224))

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20,
                             horizontal_flip=True)

# Loading the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(include_top=False,
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

# Train the model with data augmentation
batch_size = 128
epochs = 10
steps_per_epoch = len(X_train) // batch_size
model.fit(datagen.flow(X_train_resized, Y_train,
          batch_size=batch_size),
          steps_per_epoch=steps_per_epoch,
          epochs=epochs)

# Save the trained model
model.save('cifar10_model.h5')
print("Trained model saved.")
