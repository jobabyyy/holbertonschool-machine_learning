#!/usr/bin/env python3
"""Python script for Time Series Model"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns


# Initialize logging
logging.basicConfig(filename='preprocessing.log', level=logging.INFO)

# Load the dataset
def load_dataset(filename):
    try:
        df = pd.read_csv(filename)
        logging.info(f'Successfully loaded {filename}')

        # Visualization of BTC Closing Prices
        sns.distplot(df['Close'])
        plt.title('Distribution of BTC Closing Prices')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

        return df
    except Exception as e:
        logging.error(f'Error loading {filename}: {e}')
        return None

# Preprocess the data
def preprocess_data(df, sequence_length=24, target_column='Close'):
    try:
        # Drop unnecessary columns (keep only timestamp and close price)
        df = df[['Timestamp', 'Close']]

        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

        # Set timestamp as the index
        df.set_index('Timestamp', inplace=True)

        # Resample data to hourly intervals and fill missing values with forward fill
        df = df.resample('H').ffill()

        # Normalize data using Min-Max scaling
        scaler = MinMaxScaler()
        df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        # Prepare sequences for training
        sequences, targets = [], []
        for i in range(len(df) - sequence_length):
            sequences.append(df.iloc[i:i + sequence_length].values)
            targets.append(df.iloc[i + sequence_length][target_column])

        return np.array(sequences), np.array(targets)
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        return None, None

# Save preprocessed data (Optional)
def save_data(sequences, targets, output_filename):
    try:
        # Flatten the sequences
        flattened_sequences = sequences.reshape(len(sequences), -1)

        # Create DataFrames
        df_sequences = pd.DataFrame(flattened_sequences)
        df_targets = pd.DataFrame(targets)

        # Concatenate sequences and targets
        df_preprocessed = pd.concat([df_sequences, df_targets], axis=1)

        # Save to CSV
        df_preprocessed.to_csv(output_filename, index=False)
        logging.info(f'Saved preprocessed data to {output_filename}')
    except Exception as e:
        logging.error(f'Error saving preprocessed data: {e}')

if __name__ == "__main__":
    input_filename = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    output_filename = 'preprocessed_data.csv'
    sequence_length = 24

    data = load_dataset(input_filename)
    if data is not None:
        sequences, targets = preprocess_data(data, sequence_length)

        if sequences is not None and targets is not None:
            # Save preprocessed data (optional)
            save_data(sequences, targets, output_filename)
