import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("plant_disease_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    class_names = ['Healthy', 'Powdery', 'Rust']
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Prediction: {predicted_class} ({confidence:.2f})")

predict_image("9c1e3a3aa68c7971.jpg")
