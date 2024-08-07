from __future__ import division, print_function

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

def model_predict(img_path):

    img = cv2.imread(img_path)

    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    image = np.array(RGBImg) / 255.0
    new_model = tf.keras.models.load_model("CNN.h5")
    predict = new_model.predict(np.array([image]))
    print(predict)
    per = np.argmax(predict, axis=1)
    print(per[0])
    print(diagnosis_dict[per[0]])
    

    return diagnosis_dict[per[0]]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/performance', methods=['GET'])
def performance():
    return render_template('performance.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        return model_predict(file_path)

    return None


if __name__ == '__main__':
    app.run(debug=True)

