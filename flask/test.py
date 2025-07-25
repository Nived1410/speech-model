from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Enable CORS (allow JS frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Or put your domain/address here for stricter security
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define your request body structure
class InputData(BaseModel):
    input: list  # Should be a nested list matching your model’s input shape

# Load your trained model (ensure the .h5 file is in the same folder or provide path)
model = load_model('my_model.h5')

@app.post("/predict")
def predict(data: InputData):
    # Assumes input is a nested list: [[...], [...], ...]
    x = np.array(data.input).reshape(1, 40, 13)  # Adjust shape as per your model
    preds = model.predict(x)
    pred_class = int(np.argmax(preds, axis=1)[0])
    return {
        'class': pred_class,
        'probabilities': preds[0].tolist()
    }
