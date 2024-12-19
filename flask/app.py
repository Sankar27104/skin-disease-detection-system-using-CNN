from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

# Load the trained model
model = load_model('skin_disease_detector.h5')

# Load class indices (ensure this matches your model's training setup)
class_indices = {'acne': 0, 'eczema': 1}  # Example class indices, adjust accordingly
class_labels = {v: k for k, v in class_indices.items()}  # Reverse the dictionary for label mapping

# Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Prediction function
def predict_skin_disease(image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        # Check confidence level for validation
        if confidence < 0.5:
            return "Invalid input. Please provide a valid skin disease image."

        disease_name = class_labels.get(class_index, "give me correct input image")
        return f"Predicted Disease: {disease_name} with confidence: {confidence:.2f}"

    except Exception as e:
        return f"Error processing the image: {str(e)}"

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        file_path = os.path.join('static/images', file.filename)
        file.save(file_path)

        # Predict the disease using the uploaded image
        result = predict_skin_disease(file_path)
        return render_template('index.html', prediction=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
