import matplotlib.pyplot as plt
import pickle
from src.data_loader import get_clean_data

def show_prediction():
    # Load model and data
    with open("models/digit_model.pkl", "rb") as f:
        model = pickle.load(f)
    X, y = get_clean_data()

    # Pick a random sample (e.g., index 500)
    sample_idx = 10001
    image = X[sample_idx].reshape(1, -1)
    prediction = model.predict(image)

    # Display the image and the model's guess
    plt.imshow(X[sample_idx].reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {y[sample_idx]} | Model Guessed: {prediction[0]}")
    plt.show()

if __name__ == "__main__":
    show_prediction()