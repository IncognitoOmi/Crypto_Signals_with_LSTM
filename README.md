# Cryptocurrency Price Prediction

This project aims to predict the price of Bitcoin (BTC) against the US Dollar (USD) using historical data and a Long Short-Term Memory (LSTM) neural network. The model is built using TensorFlow and Keras, and it leverages data from Yahoo Finance.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data](#data)
- [Training](#training)
- [Testing](#testing)
- [Prediction](#prediction)
- [Results](#results)
- [License](#license)

## Introduction

This project uses historical price data of Bitcoin (BTC) against the US Dollar (USD) to train an LSTM neural network. The model predicts future prices based on past data, aiming to capture the temporal dependencies in the time series data.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Pandas
- pandas_datareader
- scikit-learn
- TensorFlow
- Keras

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/crypto-price-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd crypto-price-prediction
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the `main.py` script:
    ```bash
    python main.py
    ```
2. The script will fetch historical BTC-USD data from Yahoo Finance, train the LSTM model, and plot the actual vs. predicted prices.

## Model Architecture

The LSTM model consists of the following layers:

- 3 LSTM layers with 50 units each and `return_sequences` set to True for the first two layers.
- Dropout layers with a rate of 0.2 to prevent overfitting.
- A Dense layer with 1 unit to output the final prediction.

## Data

The data is fetched from Yahoo Finance using `pandas_datareader`. The script fetches data from January 1, 2018, to the current date.

## Training

The training data is prepared by scaling the closing prices between 0 and 1 using `MinMaxScaler`. The model is trained on sequences of 5 days of closing prices to predict the price on the 6th day.

## Testing

The model is tested on data from January 1, 2020, to the current date. The test data is scaled and prepared similarly to the training data.

## Prediction

The model predicts future prices based on the last 5 days of closing prices from the test data. It also predicts the next day's price using the most recent data.

## Results

The script plots the actual vs. predicted prices for the test data. It also prints the predicted price for the next day.

