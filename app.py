from flask import Flask, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image  # Using Pillow for image handling
import numpy as np
import io
from flask_cors import CORS


# SUMMARY:
# This is an API that receives an image and returns 
# what is the image that was uploaded

app = Flask(__name__)
CORS(app)
# here we're loding the pre trained model for
# image clasification
model = MobileNetV2(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Use Pillow to open and process the image
    img = Image.open(io.BytesIO(file.read()))

    # Resize image to match the input size expected by MobileNet (224x224)
    img = img.resize((224, 224))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Expand dimensions to match the input shape required by the model (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image array to the format MobileNet expects
    img_array = preprocess_input(img_array)

    # Make a prediction using the pre-trained model
    preds = model.predict(img_array)

    # Decode the top 3 predictions
    decoded_preds = decode_predictions(preds, top=3)[0]

    results = [
        {
            'label': str(pred[1]),  # Class label
            'description': str(pred[1]),  # Human-readable label
            'score': float(pred[2])  # Confidence score (converted to Python float)
        }
        for pred in decoded_preds
    ]
    # return jsonify({'predictions': results})

    final_res = sorted(results, key = lambda d: d['score'], reverse=True)

    return  jsonify({'final_result': final_res[0]})

if __name__ == '__main__':
    app.run(debug=True)



