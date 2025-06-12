import onnxruntime as ort
from PIL import Image
import numpy as np
from pathlib import Path
import io


class OnnxModel:
    """
    A class to handle loading the ONNX model and running predictions.
    It encapsulates the ONNX Runtime session and the prediction logic.
    """

    def __init__(self, model_path: Path):
        """
        Initializes the OnnxModel.

        Args:
            model_path (Path): The file path to the .onnx model.
        """
        print("Initializing ONNX Runtime session...")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at the specified path: {model_path}"
            )

        # Set up the session options for GPU execution if available
        # Cerebrium provides CUDA execution providers.
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 GB
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX session initialized successfully. Input name: '{self.input_name}'")

    def predict(self, image_data: bytes) -> int:
        """
        Performs inference on a single image.

        Since preprocessing is part of the ONNX graph, this method only needs
        to load the image, convert it to a numpy array, and run the session.

        Args:
            image_data (bytes): The raw byte data of the image (e.g., from a file).

        Returns:
            int: The predicted class ID.
        """
        try:
            # Open the image from byte data
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise IOError(
                f"Could not open image data. Ensure it's a valid image format. Error: {e}"
            )

        # Convert the PIL Image to a numpy array
        # The model expects a uint8 array of shape (H, W, C)
        image_np = np.array(image, dtype=np.uint8)

        # Add a batch dimension to create a tensor of shape (1, H, W, C)
        input_tensor = np.expand_dims(image_np, axis=0)

        # Run inference
        try:
            result = self.session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            raise RuntimeError(f"ONNX session inference failed. Error: {e}")

        # The output is a list containing one numpy array with shape (1, 1000)
        probabilities = result[0]

        # Get the index of the highest probability, which is the class ID
        predicted_class_id = int(np.argmax(probabilities))

        return predicted_class_id
