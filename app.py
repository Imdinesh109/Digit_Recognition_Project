import streamlit as st
import pickle
import numpy as np
from PIL import Image

# 1. Load the trained model
@st.cache_resource
def load_model():
    with open("models/digit_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# 2. UI Header
st.title("🔢 MNIST Digit Predictor")
st.write("Upload a handwritten digit image (28x28) to see the prediction!")

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image to grayscale and resize to 28x28
    img = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(img, caption='Uploaded Image', use_column_width=False)
    
    # Preprocess: Convert to array and flatten (784 features)
    img_array = np.array(img).reshape(1, -1) / 255.0  # Normalize
    
    # 4. Predict
    if st.button('Predict'):
        prediction = model.predict(img_array)
        st.success(f"The model predicts this is a: **{prediction[0]}**")