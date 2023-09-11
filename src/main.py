# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:02:42 2023

@author: Zeki's
"""

from data_preprocessing import preprocess_data
from train_model import train_classifier
from evaluate_model import evaluate_model
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trained_model = train_classifier(X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)

    # Get user input for feature values
    alcohol = float(input("Enter alcohol content: "))
    malic_acid = float(input("Enter malic acid content: "))
    ash = float(input("Enter ash content: "))
    alcalinity_of_ash = float(input("Enter alcalinity of ash: "))
    magnesium = float(input("Enter magnesium content: "))
    total_phenols = float(input("Enter total phenols content: "))
    flavanoids = float(input("Enter flavanoids content: "))
    nonflavanoid_phenols = float(input("Enter nonflavanoid phenols content: "))
    proanthocyanins = float(input("Enter proanthocyanins content: "))
    color_intensity = float(input("Enter color intensity: "))
    hue = float(input("Enter hue: "))
    od280_od315_of_diluted_wines = float(input("Enter OD280/OD315 of diluted wines: "))
    proline = float(input("Enter proline content: "))
    # ... (collect input for other features)

    # Create a DataFrame from the user input
    user_data = pd.DataFrame({
    'alcohol': [alcohol],
    'malic_acid': [malic_acid],
    'ash': [ash],
    'alcalinity_of_ash': [alcalinity_of_ash],
    'magnesium': [magnesium],
    'total_phenols': [total_phenols],
    'flavanoids': [flavanoids],
    'nonflavanoid_phenols': [nonflavanoid_phenols],
    'proanthocyanins': [proanthocyanins],
    'color_intensity': [color_intensity],
    'hue': [hue],
    'od280/od315_of_diluted_wines': [od280_od315_of_diluted_wines],
    'proline': [proline]
})

    # Use the trained model to make predictions
    prediction = trained_model.predict(user_data)

    # Print the prediction
    print("Predicted target class:", prediction)

if __name__ == "__main__":
    main()