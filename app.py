from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)
model = load_model('model/cnn_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224)) 
    img_array = np.array(image)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file)
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]
            result = "Monkeypox Detected" if prediction > 0.5 else "Non-Monkeypox"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
