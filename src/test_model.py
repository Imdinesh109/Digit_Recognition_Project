import matplotlib.pyplot as plt
import pickle
import numpy as np
from src.data_loader import get_clean_data

def show_actual_testing():
    # 1. Load Model
    with open("models/digit_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # 2. Get Data
    X, y = get_clean_data()
    
    # 3. Pick 5 random indices to test
    indices = [0, 101, 799, 1999, 9999]
    
    plt.figure(figsize=(12, 4))
    
    for i, idx in enumerate(indices):
        image = X[idx].reshape(28, 28)
        prediction = model.predict(X[idx].reshape(1, -1))[0]
        actual = y[idx]
        
        # Create a subplot for each digit
        plt.subplot(1, 5, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Pred: {prediction}\nActual: {actual}")
        plt.axis('off')
    
    print("Close the image window to finish the script.")
    plt.show()

if __name__ == "__main__":
    show_actual_testing()