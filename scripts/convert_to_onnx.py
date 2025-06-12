import sys
import os

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(project_root)
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from pytorch_model import Classifier, BasicBlock
from pathlib import Path


class PreprocessingWrapper(nn.Module):
    """
    A wrapper to embed preprocessing steps into the ONNX model.
    This module takes a raw uint8 image tensor and prepares it for the Classifier.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Normalization constants for ImageNet
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the preprocessing pipeline.

        Args:
            x (torch.Tensor): Input tensor of shape (N, H, W, C) and dtype uint8.

        Returns:
            torch.Tensor: The output from the classifier model.
        """
        # 1. Permute from (N, H, W, C) to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # 2. Cast to float and scale to [0, 1]
        x = x.to(torch.float32) / 255.0

        # 3. Resize to 224x224 using bilinear interpolation
        # FIX: Changed antialias to False to ensure ONNX compatibility.
        x = F.resize(x, [224, 224], antialias=False)

        # 4. Normalize the image
        x = (x - self.mean) / self.std

        # 5. Pass through the original model
        return self.model(x)


def main():
    """
    Main function to load the PyTorch model, wrap it with preprocessing,
    and export it to the ONNX format.
    """
    print("Starting ONNX conversion process...")

    # Define paths
    weights_path = Path("./weights/pytorch_model_weights.pth")
    output_dir = Path("./app")
    onnx_path = output_dir / "model.onnx"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download weights if they don't exist ---
    if not weights_path.is_file():
        print(f"Weights not found at {weights_path}.")
        print("Please download 'pytorch_model_weights.pth' from:")
        print(
            "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
        )
        print(f"And place it in the '{weights_path.parent}' directory.")
        # As an alternative, attempt to download it automatically
        try:
            print("Attempting to download weights automatically...")
            weights_path.parent.mkdir(exist_ok=True)
            url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
            torch.hub.download_url_to_file(url, str(weights_path))
            print("Weights downloaded successfully.")
        except Exception as e:
            print(f"Could not automatically download weights: {e}")
            return

    # --- Step 2: Load the PyTorch model ---
    print("Loading PyTorch model and weights...")
    pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
    pytorch_model.load_state_dict(torch.load(weights_path))
    pytorch_model.eval()
    print("PyTorch model loaded successfully.")

    # --- Step 3: Wrap the model with the preprocessing module ---
    print("Wrapping model with preprocessing layer...")
    wrapped_model = PreprocessingWrapper(pytorch_model)
    wrapped_model.eval()
    print("Model wrapped successfully.")

    # --- Step 4: Export to ONNX ---
    # Create a dummy input tensor that matches the expected input format
    # Shape: (batch_size, height, width, channels), dtype: uint8
    dummy_input = torch.randint(0, 256, (1, 300, 400, 3), dtype=torch.uint8)

    print(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["class_probabilities"],
        dynamic_axes={
            "input_image": {0: "batch_size", 1: "height", 2: "width"},
            "class_probabilities": {0: "batch_size"},
        },
    )
    print("ONNX export completed successfully.")
    print(f"Model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
