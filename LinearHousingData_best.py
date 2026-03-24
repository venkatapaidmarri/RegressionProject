import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = 'housing.csv'  # Update if your data file is elsewhere

def load_data(path):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Drop duplicates
    df = df.drop_duplicates()
    # Handle missing values
    df = df.dropna()
    # Example: Select numeric columns only (customize as needed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' not in numeric_cols:
        raise ValueError("Target column 'price' not found in numeric columns.")
    features = [col for col in numeric_cols if col != 'price']
    X = df[features]
    y = df['price']
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info(f"Preprocessed data: {X.shape[1]} features")
    return X_scaled, y

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Test MSE: {mse:.2f}")
    logging.info(f"Test R^2: {r2:.2f}")
    return y_pred, mse, r2

def plot_results(y_test, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()

def main():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred, mse, r2 = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()