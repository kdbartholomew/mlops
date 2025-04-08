from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df

def preprocess_data(df):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['target']), 
        df['target'], 
        test_size=0.2, 
        random_state=42
    )

    # Calculate mean and standard deviation from training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Transform the data
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Save processed datasets
    X_train_scaled.to_csv('data/processed_train_features.csv', index=False)
    X_test_scaled.to_csv('data/processed_test_features.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/train_target.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/test_target.csv', index=False)

if __name__ == "__main__":
    df = load_data()
    preprocess_data(df)