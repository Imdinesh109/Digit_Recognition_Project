# Step 1: Loading Data
from sklearn.datasets import fetch_openml
import numpy as np

def get_clean_data():
    print("Fetching MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]

    # --- THE CLEANING LOGIC ---
    
    # 1. Normalization (Scaling 0-255 to 0-1)
    X = X / 255.0 
    
    # 2. Type Casting (Ensure labels are integers, not strings)
    y = y.astype(np.uint8)

    print("✅ Data normalized and cleaned.")
    return X, y

if __name__ == "__main__":
    get_clean_data()