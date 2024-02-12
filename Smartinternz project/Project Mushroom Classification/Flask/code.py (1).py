#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install Flask


# In[3]:


from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model = load_model('mushroom.csv')

# Define the function to process image for prediction
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize the image according to your model's input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['image']
        img_path = 'static/' + img.filename
        img.save(img_path)
        
        processed_img = process_image(img_path)
        
        # Make prediction using your loaded model
        prediction = model.predict(processed_img)
        
        # Process prediction result here and return it
        # For example, convert prediction to a human-readable label
        
        # Return prediction to the front-end
        return f"The predicted class is: {prediction}"  # Replace this with your processed prediction

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




