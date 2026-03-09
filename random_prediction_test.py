import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random

# Path to the test dataset
test_dir = '/content/drive/My Drive/PAPER/Test'

# Load the trained model
model = tf.keras.models.load_model('/content/drive/My Drive/PAPER/movement_model.h5')

# Get the list of movement folders inside the test directory
movements = os.listdir(test_dir)

# Select a random movement folder
random_movement = random.choice(movements)
movement_path = os.path.join(test_dir, random_movement)

# Select a random image from that movement
random_image_name = random.choice(os.listdir(movement_path))
random_image_path = os.path.join(movement_path, random_image_name)

# Load and preprocess the image
img = image.load_img(random_image_path, target_size=(250, 250))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize pixel values

# Make prediction
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
predicted_class = movements[predicted_class_index]

# Print results
print(f'Selected image: {random_image_name}')
print(f'Real movement: {random_movement}')
print(f'Predicted movement by the model: {predicted_class}')
