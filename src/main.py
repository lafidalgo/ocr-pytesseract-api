# https://www.youtube.com/watch?v=0TFWtfFY87U&ab_channel=NeuralNine
# https://www.youtube.com/watch?v=sXHjYjV-36I

import uvicorn
from fastapi import FastAPI
# import pickle

# Load model
# model = pickle.load(open('model_wine_quality.pkl', 'rb'))

# Instanciate FastAPI
app = FastAPI()

# Define an endpoint
@app.get("/")
def central_function():
    return {"name": "Luiz Augusto",
            "surname": "Fidalgo Dantas"}

# Define other endpoint
@app.get("/predict")
def prediction_function():
    return {"test": "Prediction"}

"""
if __name__ == "__main__":
    # Start FastAPI
    uvicorn.run(app, port=8000, host="0.0.0.0")
"""