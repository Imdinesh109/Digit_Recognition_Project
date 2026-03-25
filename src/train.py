# Step 3: Training Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.data_loader import get_clean_data
import pickle

def train_digit_model():
    # 1. Get our cleaned and normalized data
    X, y = get_clean_data()

    # 2. SPLIT the data (The Logic: Training vs Testing)
    # We use 80% for training and 20% for the final exam
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training on {len(X_train)} images. Testing on {len(X_test)} images.")

    # 3. CHOOSE ALGORITHM
    # LogisticRegression is great for multi-class classification (0-9)
    # 'max_iter' is set to 100 to keep it fast for your laptop
    model = LogisticRegression(max_iter=100, solver='liblinear')

    # 4. TRAIN (The .fit() method is where the learning happens)
    print("Model is learning... please wait...")
    model.fit(X_train, y_train)

    # 5. SAVE the model
    # We save it as a .pkl file so we don't have to retrain it every time
    with open("models/digit_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Model trained and saved in models/digit_model.pkl!")
    return model, X_test, y_test

if __name__ == "__main__":
    train_digit_model()