import os
import numpy as np
from PIL import Image
import cv2
from keras import models
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model once to prevent repeated loading
model = models.load_model('vgg16_brain_tumor_updated.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define class names
CLASS_NAMES = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}

# Ensure 'uploads' folder exists
uploads_folder = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)

def get_class_name(class_index):
    """Maps the class index to its corresponding name."""
    return CLASS_NAMES.get(class_index, "Unknown Class")

def get_result(img_path, model):
    """Preprocess the image and predict its class."""
    image = cv2.imread(img_path)

    # Ensure the image is loaded correctly
    if image is None:
        print(f"Error: Unable to load image from path {img_path}")
        return None, None

    # Convert to RGB and resize to match input size of the model
    img = Image.fromarray(image, 'RGB')
    img = img.resize((128,128))  # VGG16 input size is 224x224

    # Normalize image to [0, 1]
    img = np.array(img) / 255.0

    # Expand dimensions to match input shape: (1, 224, 224, 3)
    input_img = np.expand_dims(img, axis=0)

    # Predict the class probabilities
    predictions = model.predict(input_img)

    # Get the index of the class with the highest probability
    predicted_class = np.argmax(predictions[0])
    
    # Get the confidence (highest probability)
    confidence = predictions[0][predicted_class] * 100

    print(f"Prediction: {predictions}")
    print(f"Predicted Class Index: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    return predicted_class, confidence

# Route to render index.html (assuming it exists in your templates folder)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']
        
        # Save the file to a secure location
        file_path = os.path.join(uploads_folder, secure_filename(f.filename))
        f.save(file_path)
        
        # Get prediction result (class index and confidence)
        class_index, confidence = get_result(file_path, model)
        
        # Map class index to class name
        result = get_class_name(class_index)
        
        # Return the result along with the confidence percentage
        return f"The predicted tumor type is: {result} with {confidence:.2f}% confidence"

    return None

if __name__ == '__main__':
    app.run(debug=True)
