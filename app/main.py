import sys
import os
from pathlib import Path
import base64
from fastapi import FastAPI, Request
from .model import OnnxModel  # Assuming model.py is in the same 'app' directory

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(project_root)

# Initialize the FastAPI app
app = FastAPI(
    title="Image Classifier API",
    description="A deep learning model to classify images using ONNX Runtime.",
    version="1.0.0",
)


class Predictor:
    def __init__(self, config):
        """
        The __init__ method is called once when the model is loaded onto a worker.
        This is where you should load your model and any other resources that are
        needed for prediction.

        Args:
            config (dict): A dictionary containing the configuration for the model.
                           (Not directly used in this simple example but can be for more complex setups)
        """
        print("Initializing predictor...")
        # The model.onnx file should be available in the /app directory inside the container
        model_path = Path("model.onnx")

        # Initialize our ONNX model handler. This will load the model into memory.
        # This one-time setup reduces latency on subsequent prediction calls.
        self.model = OnnxModel(model_path)
        print("Predictor initialized successfully.")

    def predict(self, item: dict) -> dict:
        """
        The predict method is called for each prediction request.

        Args:
            item (dict): The request payload from the user. We expect it to
                         contain a base64-encoded image string.

        Returns:
            dict: A dictionary containing the prediction result.
        """
        # --- 1. Validate Input ---
        if "image" not in item:
            return {"error": "Request payload must contain an 'image' field."}

        image_b64 = item["image"]
        if not isinstance(image_b64, str):
            return {"error": "The 'image' field must be a base64-encoded string."}

        # --- 2. Decode Image ---
        try:
            # Decode the base64 string to bytes
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            return {"error": f"Failed to decode base64 string. Error: {e}"}

        # --- 3. Run Prediction ---
        try:
            # Pass the raw image bytes to our model's predict method
            predicted_class_id = self.model.predict(image_bytes)
        except (IOError, RuntimeError) as e:
            # Catch potential errors from the model class (e.g., bad image data)
            return {"error": str(e)}

        # --- 4. Format and Return Output ---
        # The prediction was successful, return the class ID.
        return {"class_id": predicted_class_id}


# Instantiate the predictor globally so it's loaded once
# In a Cerebrium custom runtime, this 'Predictor' class is what Cerebrium expects.
# It will call Predictor().__init__(config) and then Predictor().predict(item).
# We are integrating it with FastAPI for the health check.
predictor_instance = None


@app.on_event("startup")
async def startup_event():
    """
    Handler for application startup event.
    Initializes the Predictor when the FastAPI application starts.
    """
    global predictor_instance
    # In a real Cerebrium deployment, 'config' might be passed here.
    # For local FastAPI testing, we'll pass an empty dict or mock config.
    predictor_instance = Predictor(config={})
    print("FastAPI startup: Predictor initialized.")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for Cerebrium.
    Returns a 200 OK if the application is running.
    """
    if predictor_instance and predictor_instance.model:
        return {"status": "ok", "message": "Model is loaded and ready"}
    return {"status": "initializing", "message": "Model is still loading"}, 503


@app.post("/predict")
async def predict_endpoint(request: Request):
    """
    Prediction endpoint that takes a JSON payload with a base64-encoded image.
    """
    if not predictor_instance:
        return {"error": "Model is not yet initialized"}, 503

    try:
        item = await request.json()
        result = predictor_instance.predict(item)
        if "error" in result:
            return result, 400  # Bad request if prediction returns an error
        return result
    except Exception as e:
        return {"error": f"An unexpected error occurred during prediction: {e}"}, 500
