import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt

# Paths to dataset
train_dir = r'C:/Users/ANJANI/Documents/Brain_Tumor_Minor_Project/BT_Dataset/Training'
test_dir = r'C:/Users/ANJANI/Documents/Brain_Tumor_Minor_Project/BT_Dataset/Testing'

# Image dimensions and batch size
IMAGE_SIZE = 128
batch_size = 32  # Updated to 32 for better GPU utilization

# Data generators for training and testing
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2,  # Reserve 20% for validation
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

# Training and validation generators
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical',  # Updated to categorical
    subset='training',  # Use 80% for training
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical',  # Updated to categorical
    subset='validation',  # Use 20% for validation
    shuffle=True
)

# Testing generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='categorical',  # Updated to categorical
    shuffle=False  # No shuffling for evaluation
)

# Map class indices to ensure compatibility with testing
class_indices = train_generator.class_indices
class_indices_reversed = {v: k for k, v in class_indices.items()}
print("Class Indices:", class_indices)

# VGG16 model setup
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Build the model
inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)  # Prevent updating base model weights
x = GlobalAveragePooling2D()(x)         # Replace Flatten with GAP
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs, outputs)
# model = Sequential([
#     Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#     base_model,
#     GlobalAveragePooling2D(),  # Alternative to Flatten (more robust)
#     Dropout(0.3),
#     Dense(256, activation='relu'),
#     Dropout(0.2),
#     Dense(len(train_generator.class_indices), activation='softmax')  # Adjust output size
# ])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',  # Updated to categorical_crossentropy
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
import os

# Correct and absolute file path
save_path = os.path.join(os.getcwd(), 'vgg16_brain_tumor_updated.keras')

checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss')

# Train the model
steps_per_epoch = train_generator.samples // batch_size
validation_steps = val_generator.samples // batch_size

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Increased epochs for better training
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1
)

# Save the model
model.save('vgg16_brain_tumor_updated.h5')

# Save class indices for testing compatibility
import json
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

# Evaluate the model on the test set
test_steps = test_generator.samples // batch_size
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.summary()