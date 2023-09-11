# Wine Quality Classifier

## Overview

Welcome to the Wine Quality Classifier project! This Python application predicts the quality of wine based on chemical attributes. It consists of several modules that handle data preprocessing, model training, evaluation, and a user interface for making predictions.

## Project Structure

- **Data Preprocessing (`data_preprocessing.py`):** This module loads and preprocesses the dataset, preparing it for training.

- **Model Training (`train_model.py`):** This module trains the machine learning classifier on the preprocessed data and saves the model.

- **Model Evaluation (`evaluate_model.py`):** This module assesses the model's performance using test data, computing metrics like accuracy, precision, recall, and F1-score.

- **User Interface (`main.py`):** This component allows users to input wine attributes for prediction.

## Usage

Follow these steps to use the Wine Quality Classifier:

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/wine_quality.git
    cd wine_quality
    cd src
    ```

2. Run the classifier:

    ```bash
    python main.py
    ```

3. Enter wine attributes as prompted, and the classifier will predict the wine quality.

## Dataset
#### Wine Recognition Data

The Wine Recognition dataset used in this project was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine).
The dataset contains chemical attributes and quality ratings of wines.

## Dependencies

This project relies on these Python libraries:

- pandas
- scikit-learn

Install them with:

```bash
pip install pandas scikit-learn
