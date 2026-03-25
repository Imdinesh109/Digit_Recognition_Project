import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.data_loader import get_clean_data
from sklearn.model_selection import train_test_split

def run_evaluation():
    # 1. Load data and model
    X, y = get_clean_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with open("models/digit_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. Make predictions on the "Exam" (Test) data
    y_pred = model.predict(X_test)

    # 3. Calculate Results
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- MODEL REPORT CARD ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed breakdown per digit:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_evaluation()