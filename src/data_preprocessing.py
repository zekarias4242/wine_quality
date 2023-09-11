
import pandas as pd

def preprocess_data():
    data = pd.read_csv('../data/wine.data', header=None)
    X = data.iloc[:, 1:]  # Features
    y = data.iloc[:, 0]   # Target

    # Perform any necessary preprocessing (scaling, encoding, etc.)
    # Return X and y
    return X, y
# Call the function if this script is executed
if __name__ == "__main__":
    X, y = preprocess_data()
