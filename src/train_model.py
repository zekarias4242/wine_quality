# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:03:24 2023

@author: Zeki's
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from data_preprocessing import preprocess_data
import joblib

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC()
    model.fit(X_train, y_train)
    
    return model

# Call the function if this script is executed
if __name__ == "__main__":
    X, y = preprocess_data()
    trained_model = train_classifier(X, y)
    joblib.dump(trained_model, '../src/trained_model.pkl')