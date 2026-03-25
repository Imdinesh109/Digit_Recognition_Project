# MNIST Digit Recognition System
An end-to-end Machine Learning pipeline developed during the Slash Mark Internship. This project classifies handwritten digits (0-9) from the MNIST dataset using a Logistic Regression model.

## 🚀 Project Overview
- **Algorithm:** Logistic Regression (One-vs-Rest)
- **Dataset:** MNIST (70,000 images, 28x28 pixels)
- **Accuracy:** ~82% on 10,000 unseen test images
- **Tools:** Python, Scikit-Learn, NumPy, Matplotlib

## 📂 Project Structure
- `src/`: Core logic (Data loading, training, evaluation)
- `models/`: Serialized `.pkl` model files
- `visualize.py`: Script to see visual predictions

## 🛠️ How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python -m src.train`
3. Test accuracy: `python -m src.test_model`
4. Visualize results: `python -m src.visualize`

## 📊 Evaluation
The model was validated using a Confusion Matrix to analyze classification errors between similar digits like 4 and 9.