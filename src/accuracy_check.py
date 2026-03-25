import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from src.data_loader import get_clean_data
from sklearn.model_selection import train_test_split

def check_detailed_accuracy():
    # 1. Load Model & Data
    with open("models/digit_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    X, y = get_clean_data()
    # Isolate the 10,000 test images (approx 14.28%)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.1428, random_state=42)

    # 2. Get Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 3. Create Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 4. Plot the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Final Test Accuracy: {acc*100:.2f}%')
    plt.show()

if __name__ == "__main__":
    check_detailed_accuracy()