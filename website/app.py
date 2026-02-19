from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("../models/spider_model.h5")
classes = ["Clean Web", "Damaged Web", "No Web"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        img = Image.open(request.files["image"])
        img = img.resize((128,128))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        prediction = classes[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
