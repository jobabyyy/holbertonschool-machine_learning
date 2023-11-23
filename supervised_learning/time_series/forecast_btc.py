#!/usr/bin/env python3
"""Python script for Time Forecast Model."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

# Initialize logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Load preprocessed data
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        logging.info(f'Successfully loaded {filename}')
        return df
    except Exception as e:
        logging.error(f'Error loading {filename}: {e}')
        return None

# Prepare sequences for training
def prepare_sequences(data, sequence_length=24, target_column='Close'):
    sequences, targets = [], []
    
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i+sequence_length].values)
        targets.append(data.iloc[i+sequence_length][target_column])
    
    return np.array(sequences), np.array(targets)

# Create an RNN model
def create_model(input_shape, rnn_units=64):
    model = keras.Sequential([
        layers.LSTM(rnn_units, return_sequences=True, input_shape=input_shape),
        layers.LSTM(rnn_units, return_sequences=False),
        layers.Dense(1)  # Output layer with 1 neuron for regression
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Testing and Evaluation
def evaluate_model(model, X_test, y_test):
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

# Prediction and performance visualization function
def plot_predictions(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Prices')
    plt.plot(y_pred.flatten(), label='Predicted Prices', alpha=0.7)  # Flatten y_pred to ensure it's 1D
    plt.title('BTC Price Prediction')
    plt.xlabel('Time (index)')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()
    
    # Return the metrics if needed
    return mse, mae, rmse

if __name__ == "__main__":
    data_filename = 'preprocessed_data.csv'
    sequence_length = 24

    # Load and prepare the data
    data = load_data(data_filename)
    if data is not None:
        sequences, targets = prepare_sequences(data, sequence_length)

        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(sequences, targets, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Convert the numpy arrays into tf.data.Dataset
        batch_size = 64
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        # Create the model
        model = create_model(X_train.shape[1:])

        # Implement early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # Train the model
        history = model.fit(train_data, validation_data=val_data, epochs=100, callbacks=[es])

        # Evaluate the model on the test dataset
        evaluate_model(model, X_test, y_test)

        # Plot training & validation loss values
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # After training and evaluation, plot predictions
        plot_predictions(model, X_test, y_test)
