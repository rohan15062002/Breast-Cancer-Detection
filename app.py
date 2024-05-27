# Importing libraries
from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import io
import base64

# OS Environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setting up Flask Application
app = Flask(__name__,static_folder='Static')

# Loading model to backend

model = load_model(r'C:\Users\LENOVO\OneDrive\Desktop\flask app\project files\Model\model.h5')


# Routing Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Routing Classify page
@app.route('/classify')
def classify():
    return render_template('classify.html')

# Backend Model prediction using api

threshold = 0.7  # Set the threshold value

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file'].read()
    img = Image.open(io.BytesIO(img))
    img = img.resize((50, 50))  # Resize to match the model's input shape
    img_array = np.array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    
    class_names = ['Benign', 'Malignant']
    predicted_class = 'Malignant' if pred[1] > threshold else 'Benign'
    probabilities = {class_names[i]: float(pred[i]) for i in range(len(class_names))}

    
    return render_template('predict.html', predicted_class=predicted_class, probabilities=probabilities)



# Routing Team page
@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

# Routing About page
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')





# Running Flask Application in host = 127.0.0.1 port = 5000
if __name__ == '__main__':
    app.run(host = '127.0.0.1',port = 5000, debug = False)