import requests
import base64
import os
import argparse
from pathlib import Path
import time
import json


def get_credentials():
    """
    Retrieves the API endpoint and key from environment variables.
    """
    endpoint = os.getenv("CEREBRIUM_ENDPOINT")
    api_key = os.getenv("CEREBRIUM_API_KEY")

    if not endpoint or not api_key:
        print("=" * 60)
        print("ERROR: Environment variables not set.")
        print("Please set CEREBRIUM_ENDPOINT and CEREBRIUM_API_KEY.")
        print("Example:")
        print("  export CEREBRIUM_ENDPOINT='https://run.cerebrium.ai/v3/p-...'")
        print("  export CEREBRIUM_API_KEY='private-...'")
        print("=" * 60)
        exit(1)

    return endpoint, api_key


def encode_image(image_path: Path) -> str:
    """
    Reads an image file and encodes it as a base64 string.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_prediction_request(
    endpoint: str, api_key: str, image_b64: str
) -> (dict, float):
    """
    Sends a prediction request to the Cerebrium endpoint.
    Returns the JSON response and the request latency.
    """
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"image": image_b64})

    start_time = time.time()
    response = requests.post(endpoint, headers=headers, data=payload)
    latency = time.time() - start_time

    return response.json(), latency


def run_single_prediction(image_path: Path):
    """
    Runs a single prediction for a given image path.
    """
    print(f"--- Running single prediction for: {image_path.name} ---")
    endpoint, api_key = get_credentials()

    try:
        image_b64 = encode_image(image_path)
        result, latency = make_prediction_request(endpoint, api_key, image_b64)

        print(f"Request Latency: {latency:.2f} seconds")
        if "class_id" in result:
            print(f"Predicted Class ID: {result['class_id']}")
        else:
            print(f"Received an error: {result}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def run_preset_tests():
    """
    Runs a suite of automated tests against the deployed endpoint.
    """
    print("\n--- Running Preset Test Suite ---")
    endpoint, api_key = get_credentials()

    test_images = {
        "tench": {"path": Path("dataset/n01440764_tench.jpeg"), "id": 0},
        "turtle": {"path": Path("dataset/n01667114_mud_turtle.JPEG"), "id": 35},
    }

    # Test 1: Correctness checks
    print("\n[1] Running Correctness Tests...")
    for name, data in test_images.items():
        try:
            image_b64 = encode_image(data["path"])
            result, _ = make_prediction_request(endpoint, api_key, image_b64)
            predicted_id = result.get("class_id", -1)
            assert (
                predicted_id == data["id"]
            ), f"Expected {data['id']}, got {predicted_id}"
            print(f"  - PASSED: '{name}' correctly classified as {predicted_id}.")
        except Exception as e:
            print(f"  - FAILED: '{name}' test failed. Error: {e}")

    # Test 2: Latency check
    print("\n[2] Running Latency Test...")
    latencies = []
    image_b64 = encode_image(test_images["tench"]["path"])
    for i in range(3):
        _, latency = make_prediction_request(endpoint, api_key, image_b64)
        latencies.append(latency)
        print(f"  - Request {i+1}/3 latency: {latency:.2f}s")
    avg_latency = sum(latencies) / len(latencies)
    print(f"  - Average latency over 3 requests: {avg_latency:.2f}s")

    # Test 3: Malformed request
    print("\n[3] Running Malformed Request Test...")
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    # Sending a payload without the 'image' key
    malformed_payload = json.dumps({"wrong_key": "some_value"})
    response = requests.post(endpoint, headers=headers, data=malformed_payload)
    if response.status_code == 200 and "error" in response.json():
        print(
            f"  - PASSED: Server correctly handled malformed payload with error: {response.json()['error']}"
        )
    else:
        print(
            f"  - FAILED: Server responded with status {response.status_code}. Expected an error message."
        )

    print("\n--- Test Suite Finished ---")


def main():
    """
    Main function to parse arguments and run the appropriate test.
    """
    parser = argparse.ArgumentParser(
        description="Test client for the Cerebrium image classifier."
    )
    parser.add_argument(
        "--image_path", type=Path, help="Path to the image for a single prediction."
    )
    parser.add_argument(
        "--run-tests", action="store_true", help="Run the full suite of preset tests."
    )

    args = parser.parse_args()

    if args.run_tests:
        run_preset_tests()
    elif args.image_path:
        run_single_prediction(args.image_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
