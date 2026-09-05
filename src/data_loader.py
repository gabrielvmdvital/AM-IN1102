import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_spambase_data(data_path="data/spambase.data"):
    """
    Loads the spambase dataset, separates features and target,
    and applies standard scaling to the features.
    """
    # The dataset has no header. The last column is the target (spam or not).
    # There are 57 features and 1 target column.
    
    # Load dataset
    df = pd.read_csv(data_path, header=None)
    
    # Separate features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == "__main__":
    X, y = load_spambase_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y class distribution: {pd.Series(y).value_counts().to_dict()}")
