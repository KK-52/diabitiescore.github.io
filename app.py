from flask import Flask, request, jsonify  
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)  


upload_dir = r'D:\uploads\uploads'  
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

model_path = r'D:\path_to_your_model\your_model.keras'   
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 200  
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 200  
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
    
        try:
            
            img = image.load_img(file_path, target_size=(200, 200))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  
            
        
            prediction = model.predict(img_array)
            result = "Low chance of diabetes" if prediction[0] > 0.5 else "High chance of diabetes"
            
            
            os.remove(file_path)
            
            return jsonify({'result': result})  
        except Exception as e:
            return jsonify({'error': str(e)}), 500  
    else:
        return jsonify({'error': 'File not allowed'}), 400  
if __name__ == '__main__':
    app.run(debug=True, host='172.16.23.25', port=5000)

