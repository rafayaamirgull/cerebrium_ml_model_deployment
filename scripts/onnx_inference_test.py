import onnxruntime as ort
from PIL import Image
import numpy as np
import argparse
from pathlib import Path


def run_inference(model_path: Path, image_path: Path):
    """
    Loads an ONNX model and runs inference on a single image.

    Args:
        model_path (Path): Path to the .onnx model file.
        image_path (Path): Path to the input image file.
    """
    # 1. Check if files exist
    if not model_path.is_file():
        print(f"Error: ONNX model not found at '{model_path}'")
        print("Please run 'python scripts/convert_to_onnx.py' first.")
        return

    if not image_path.is_file():
        print(f"Error: Image not found at '{image_path}'")
        return

    print(f"Loading model: {model_path}")
    print(f"Loading image: {image_path}")

    # 2. Set up ONNX Runtime session
    # Using CPUExecutionProvider for this simple script, as it doesn't require a GPU.
    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX session: {e}")
        return

    # 3. Load and prepare the image
    try:
        # Open the image and ensure it's in RGB format
        image = Image.open(image_path).convert("RGB")

        # Convert the PIL Image to a numpy array (H, W, C) with dtype uint8
        image_np = np.array(image, dtype=np.uint8)

        # Add a batch dimension to create a tensor of shape (1, H, W, C)
        input_tensor = np.expand_dims(image_np, axis=0)

    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 4. Run inference
    print("Running inference...")
    try:
        result = session.run(None, {input_name: input_tensor})

        # The output is a list containing one numpy array with shape (1, 1000)
        probabilities = result[0]

        # Get the index of the highest probability, which is the class ID
        predicted_class_id = int(np.argmax(probabilities))

        print("\n--- Inference Complete ---")
        print(f"Predicted Class ID: {predicted_class_id}")

    except Exception as e:
        print(f"An error occurred during inference: {e}")


def main():
    """
    Parses command-line arguments to get the model and image paths.
    """
    parser = argparse.ArgumentParser(
        description="Run inference on an ONNX image classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("app/model.onnx"),
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image file (e.g., dataset/n01440764_tench.JPEG).",
    )
    args = parser.parse_args()

    run_inference(args.model, args.image)


if __name__ == "__main__":
    main()
