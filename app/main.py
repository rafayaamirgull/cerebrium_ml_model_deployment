import sys
import os

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(project_root)
from cerebrium import get_secret
from pathlib import Path
import base64
import io

# It's good practice to place the model loading logic in a separate module
# to keep the main application file clean.
from .model import OnnxModel


class Predictor:
    def __init__(self, config):
        """
        The __init__ method is called once when the model is loaded onto a worker.
        This is where you should load your model and any other resources that are
        needed for prediction.

        Args:
            config (dict): A dictionary containing the configuration for the model.
        """
        print("Initializing predictor...")
        model_path = Path("model.onnx")  # The model is in the same directory

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
