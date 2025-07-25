from flask import Flask, request, jsonify
import numpy as np
import librosa
import tempfile
import os

# Keras import for model loading
from keras.models import load_model

app = Flask(__name__)

# === LOAD YOUR MODEL AT THE START ===
# Absolute path (best): 
model_path = r'C:\Users\ADMIN\Downloads\WT-Record\flask\my_model.keras'
# Or, if the script is in the same folder, just use 'my_model.keras'
model = load_model(model_path)

def extract_features_or_melspectrogram(y, sr):
    # Implement your actual feature extraction logic here!
    # EXAMPLE:
    # returns np.expand_dims(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=0)
    # Replace with what your model expects!
    return np.zeros((1, 40, 174, 1))  # dummy shape example

def predict_emotion(file_path):
    y, sr = librosa.load(file_path, sr=None)
    X = extract_features_or_melspectrogram(y, sr)  # Shape must match model input
    prediction = model.predict(X)
    # Map prediction to emotion label (adjust as per your model logic)
    emotion_index = np.argmax(prediction)
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']  # your actual classes!
    emotion = emotion_labels[emotion_index]
    confidence = float(np.max(prediction))
    return {"emotion": emotion, "confidence": confidence}

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400
    f = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpf:
        f.save(tmpf)
        tmp_path = tmpf.name
    result = predict_emotion(tmp_path)
    os.unlink(tmp_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
