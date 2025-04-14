# Disaster Tweet Classification

This project implements a machine learning model to classify tweets as either related to a real disaster (positive) or not (negative) using natural language processing (NLP) techniques. It leverages the `all-MiniLM-L6-v2` model from the `sentence-transformers` library to generate text embeddings and a simple neural network built with TensorFlow for binary classification.

## Overview
The goal of this project is to predict whether a tweet describes a real disaster event based on its text content. The notebook processes a dataset of labeled tweets, generates embeddings using a pre-trained transformer model, trains a neural network, and generates predictions for a test dataset.

Key steps include:
- Loading and preprocessing tweet data.
- Generating embeddings using `all-MiniLM-L6-v2`.
- Training a neural network for binary classification.
- Making predictions on a test set and saving results to a CSV file.

## Dataset
The dataset consists of two CSV files:
- `train.csv`: Contains tweet IDs, text, keywords, locations, and target labels (1 for disaster, 0 for non-disaster).
- `test.csv`: Contains tweet IDs, text, keywords, and locations for predictions.

The dataset is assumed to be in the same directory as the notebook. The training data has 7,613 samples, and the test data has 3,263 samples.

## Installation
To run this project, you need Python 3.12 or later and the following dependencies:

```bash
pip install transformers sentence-transformers tensorflow numpy pandas scikit-learn
