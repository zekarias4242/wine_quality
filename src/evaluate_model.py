# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:13:23 2023

@author: Zeki's
"""

from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)

# Call the function if this script is executed
if __name__ == "__main__":
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trained_model = train_classifier(X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)