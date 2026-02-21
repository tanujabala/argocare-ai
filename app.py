from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import os

# -------------------- App Setup --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploadimages"
MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- Load Model --------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------- Load Disease Info --------------------
with open("plant_disease.json", "r") as f:
    plant_disease = json.load(f)

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/uploadimages/<path:filename>")
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -------------------- Image Processing --------------------
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return plant_disease[class_index]

# -------------------- Upload & Predict --------------------
@app.route("/upload/", methods=["POST"])
def upload_image():
    if "img" not in request.files:
        return redirect("/")

    image = request.files["img"]
    if image.filename == "":
        return redirect("/")

    filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    prediction = predict_disease(image_path)

    return render_template(
        "home.html",
        result=True,
        imagepath=f"/uploadimages/{filename}",
        prediction=prediction
    )

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)