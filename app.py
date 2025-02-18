from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy
import os
from geminiResponse import get_instructions

app = Flask(__name__)

MODEL_PATh = "plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATh)

class_names = ["Healthy", "Powdery", "Rust"]

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = numpy.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = "temp.jpg"
    file.save(file_path)

    img_array = preprocess_image(file_path)

    predictions = model.predict(img_array)
    predicted_class = class_names[numpy.argmax(predictions)]
    confidence = float(numpy.max(predictions))
    instructions = get_instructions("Tomato", predicted_class)

    os.remove(file_path)

    return jsonify({"class": predicted_class, "confidence": confidence, "instructions": instructions})

if __name__ == "__main__":
    app.run(debug=True)
