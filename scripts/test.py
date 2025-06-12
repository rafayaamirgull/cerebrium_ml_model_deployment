import pytest
from pathlib import Path
import os
import sys

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(project_root)
# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model import OnnxModel


@pytest.fixture(scope="module")
def onnx_model():
    """
    Pytest fixture to initialize the OnnxModel once for all tests in this module.
    This saves time by not reloading the model for every single test case.
    """
    model_path = Path("app/model.onnx")
    if not model_path.exists():
        pytest.fail(
            "model.onnx not found. Please run 'python scripts/convert_to_onnx.py' first."
        )
    return OnnxModel(model_path)


@pytest.fixture
def image_paths():
    """
    Provides a dictionary of test images and their expected class IDs.
    """
    dataset_dir = Path("./dataset")
    if not dataset_dir.is_dir():
        pytest.fail(
            "Dataset directory not found. Please create it and add test images."
        )
    return {
        "tench": {"path": dataset_dir / "n01440764_tench.jpeg", "id": 0},
        "turtle": {"path": dataset_dir / "n01667114_mud_turtle.JPEG", "id": 35},
    }


def test_tench_prediction(onnx_model, image_paths):
    """
    Tests if the model correctly classifies the 'tench' image.
    """
    print("Testing 'tench' image...")
    image_info = image_paths["tench"]
    image_path = image_info["path"]
    expected_id = image_info["id"]

    if not image_path.is_file():
        pytest.fail(f"Test image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    predicted_id = onnx_model.predict(image_bytes)
    assert (
        predicted_id == expected_id
    ), f"Failed for tench. Expected {expected_id}, got {predicted_id}."
    print(f"'tench' test PASSED. Predicted: {predicted_id}, Expected: {expected_id}")


def test_turtle_prediction(onnx_model, image_paths):
    """
    Tests if the model correctly classifies the 'mud turtle' image.
    """
    print("Testing 'turtle' image...")
    image_info = image_paths["turtle"]
    image_path = image_info["path"]
    expected_id = image_info["id"]

    if not image_path.is_file():
        pytest.fail(f"Test image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    predicted_id = onnx_model.predict(image_bytes)
    assert (
        predicted_id == expected_id
    ), f"Failed for turtle. Expected {expected_id}, got {predicted_id}."
    print(f"'turtle' test PASSED. Predicted: {predicted_id}, Expected: {expected_id}")


def test_invalid_image_input(onnx_model):
    """
    Tests if the model prediction call handles corrupted/invalid image data gracefully.
    It should raise an IOError.
    """
    print("Testing invalid image input...")
    invalid_data = b"this is not an image"
    with pytest.raises(IOError) as excinfo:
        onnx_model.predict(invalid_data)

    assert "Could not open image data" in str(excinfo.value)
    print("Invalid image input test PASSED.")


# To run these tests, navigate to the project root and execute:
# pytest scripts/test.py -v -s
# The '-s' flag is included to show the print statements during execution.
# The '-v' flag is included to show detailed test information.
