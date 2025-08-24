from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model  # Import load_model from keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((48, 48))

    # Convert to RGB if needed (use 'RGB' mode if the image is already RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')  

    img_array = np.array(img) / 255.0  # Normalize
    return img_array

def predict_emotion(image_array):
    global best_model  # Access the global best_model variable
    prediction = best_model.predict(image_array[np.newaxis, ...])
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral' 
    }
    return emotion_labels[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_array = load_and_preprocess_image(image_path)
            predicted_emotion = predict_emotion(image_array)
            # Send the actual filename back in the response
            return jsonify({'emotion': predicted_emotion, 'filename': filename}) 
        else:
            return jsonify({'error': 'Invalid file type'})
    else:
        return render_template('index.html')
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    global best_model
    best_model = load_model('fer_vgg.h5')
    app.run(debug=True)