import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMAGE_SIZE = 128  # Must match training dimensions

# Load the model
model_path = 'vgg16_brain_tumor_updated.h5'
model = load_model(model_path)

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_indices_reversed = {v: k for k, v in class_indices.items()}  # Reverse mapping
print("Class Indices Mapping:", class_indices_reversed)

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))  # Resize to model input size
    img_array = img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to predict class
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_indices_reversed[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class_label, confidence

# Test a single image
test_image_path = r"BT_Dataset/Testing/notumor/Te-no_0015.jpg"
if os.path.exists(test_image_path):
    predicted_label, confidence = predict_image(test_image_path)
    print(f"Predicted Class: {predicted_label} (Confidence: {confidence:.2f}%)")
else:
    print(f"Image not found at {test_image_path}")

# Optional: Test multiple images in a folder
test_folder_path = r"BT_Dataset/Testing"  # Replace with your folder path
if os.path.exists(test_folder_path):
    for img_file in os.listdir(test_folder_path):
        image_path = os.path.join(test_folder_path, img_file)
        if image_path.endswith(('.jpg', '.jpeg', '.png')):
            predicted_label, confidence = predict_image(image_path)
            print(f"Image: {img_file} --> Predicted Class: {predicted_label} (Confidence: {confidence:.2f}%)")
else:
    print(f"Folder not found at {test_folder_path}")
