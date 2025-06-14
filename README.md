# Deep Learning Image Classifier Deployment on Cerebrium

This repository contains a complete solution for deploying a pre-trained deep learning image classification model (based on ResNet18) to Cerebrium's serverless platform. The project demonstrates the full MLOps workflow, including ONNX model conversion with embedded preprocessing, building a FastAPI inference API, Docker containerization, cloud deployment, and comprehensive testing.

## Table of Contents

- [Deep Learning Image Classifier Deployment on Cerebrium](#deep-learning-image-classifier-deployment-on-cerebrium)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Prerequisites](#2-prerequisites)
  - [3. Getting Started: Local Setup \& Model Preparation](#3-getting-started-local-setup--model-preparation)
    - [3.1 Clone the Repository](#31-clone-the-repository)
    - [3.2 Install Dependencies](#32-install-dependencies)
    - [3.3 Download PyTorch Model Weights](#33-download-pytorch-model-weights)
    - [3.4 Prepare Test Dataset](#34-prepare-test-dataset)
    - [3.5 Convert PyTorch Model to ONNX](#35-convert-pytorch-model-to-onnx)
    - [3.6 Run Local Unit Tests](#36-run-local-unit-tests)
  - [4. Deployment to Cerebrium](#4-deployment-to-cerebrium)
    - [4.1 Review Configuration Files](#41-review-configuration-files)
      - [`cerebrium.toml`](#cerebriumtoml)
      - [`Dockerfile`](#dockerfile)
    - [4.2 Initiate Deployment](#42-initiate-deployment)
    - [4.3 Monitor Deployment](#43-monitor-deployment)
  - [5. Testing the Deployed Model](#5-testing-the-deployed-model)
    - [5.1 Set Environment Variables](#51-set-environment-variables)
    - [5.2 Run Preset Tests](#52-run-preset-tests)
  - [6. Code Structure](#6-code-structure)
  - [7. Key Design Decisions \& Potential Improvements](#7-key-design-decisions--potential-improvements)
    - [7.1 Design Decisions](#71-design-decisions)
    - [7.2 Potential Improvements](#72-potential-improvements)
  - [8. Assignment Completion Status](#8-assignment-completion-status)
  - [9. Loom Walkthrough Video](#9-loom-walkthrough-video)

---

## 1. Project Overview

This project provides a complete solution for deploying an image classification model to Cerebrium. It encompasses:

- **Model Conversion**: Converting a PyTorch model to ONNX format, embedding image preprocessing steps directly into the ONNX graph for efficient and consistent inference.
- **API Development**: Building a high-performance REST API using FastAPI to serve model predictions.
- **Containerization**: Defining a custom Docker image for consistent deployment across environments.
- **Cloud Deployment**: Configuring and deploying the containerized application to Cerebrium.ai, a serverless platform optimized for machine learning workloads.
- **Testing**: Providing robust testing scripts for both local model validation and end-to-end verification of the deployed endpoint, including correctness, latency, and error handling.

The model is pre-trained on the ImageNet dataset and is capable of classifying images into 1000 categories, expecting responses within 2–3 seconds in production.

## 2. Prerequisites

Before you start, ensure you have the following installed on your system:

- **Python 3.10+**
- **Git**
- **Docker** (optional but helpful)
- **Cerebrium CLI**:
  ```bash
  pip install cerebrium
  cerebrium login
  ```

- **Python Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

## 3. Getting Started: Local Setup & Model Preparation

### 3.1 Clone the Repository

```bash
git clone https://github.com/rafayaamirgull/cerebrium_ml_model_deployment
cd cerebrium_ml_model_deployment
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Download PyTorch Model Weights

```bash
mkdir -p weights
# Download weights manually from:
# https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0
# Place into weights/
```

### 3.4 Prepare Test Dataset

```bash
mkdir -p dataset
# Place these files into dataset/:
# - n01440764_tench.jpeg (class ID: 0)
# - n01667114_mud_turtle.JPEG (class ID: 35)
```

### 3.5 Convert PyTorch Model to ONNX

```bash
python scripts/convert_to_onnx.py
```

### 3.6 Run Local Unit Tests
To test the onnx converted model inference, run the following command:
```bash
pytest scripts/test.py -v -s
```

---

## 4. Deployment to Cerebrium

### 4.1 Review Configuration Files

#### `cerebrium.toml`
```toml
name = "mtailor-classifier"
python_version = "3.10"
gpu_count = 0
port = 8192
healthcheck_endpoint = "/health"
dockerfile_path = "./Dockerfile"
```

#### `Dockerfile`

```Dockerfile
FROM python:3.10-slim
WORKDIR /workspace
COPY ./ ./
RUN pip install -r requirements.txt
ENV PYTHONPATH="/workspace"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8192"]
```

### 4.2 Initiate Deployment

```bash
cerebrium deploy #You need to login first on linux terminal with "cerebrium login" command
```

### 4.3 Monitor Deployment

Check your Cerebrium dashboard for deployment status.

---

## 5. Testing the Deployed Model

### 5.1 Set Environment Variables

```bash
export CEREBRIUM_ENDPOINT="https://api.cortex.cerebrium.ai/v4/p-c4721f96/mtailor-classifier/predict"

export CEREBRIUM_API_KEY="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWM0NzIxZjk2IiwiaWF0IjoxNzQ5Mzk2OTU1LCJleHAiOjIwNjQ5NzI5NTV9.aXYFY4W5ZzFEou9I3YAmMgHzMt1OAzFNR3t0nn8GR3ET9gFQbYmOh13F4Jw_xND-_sLKyK_iY9AOSoUKX-0LCh_IfWLH_mud0YnLykV4YdLLOtBcWQSOCygaNaiOrem-IiBoSjTiHWC_ovZNdHXJZeR3G1dbzZHSCkdwo8d0Sbt_CuPVO4BWXQHA-_LbNQJ2XOlsfJqt4qF9bJiSQrvjbssaKR0OiApcwmMlqhXoF1kQbosphVozr8y1P0QW_ScXsfx7WtLFwVABrVzaKBGOAffkiYbcVOOMCLvRzZfPq7vSVcWbKFxj30ZH1SlhQIVe9Ki5Ki9N0gciLMbGe9Q9Fg"

```

### 5.2 Run Preset Tests

```bash
python scripts/test_server.py --run-tests
```
---

### 5.3 Run Single Frame Inference Test

```bash
python scripts/test_server.py  --image_path "<path-to-your-image>"
```
---

## 6. Code Structure

```
project-root/
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── model.onnx
│
├── scripts/
│   ├── convert_to_onnx.py
│   ├── onnx_inference_test.py
│   ├── test.py
│   └── test_server.py
│
├── weights/
│   └── pytorch_model_weights.pth
│
├── dataset/
│   ├── n01440764_tench.jpeg
│   └── n01667114_mud_turtle.JPEG
│
├── Dockerfile
├── cerebrium.toml
├── requirements.txt
└── pytorch_model.py
```

---

## 7. Key Design Decisions & Potential Improvements

### 7.1 Design Decisions

- **ONNX Conversion with Preprocessing**: Embedding image preprocessing directly into the ONNX graph.
- **FastAPI**: Used for its performance and developer-friendly features.
- **OnnxModel Class**: Centralized model loading and inference logic.
- **Startup Initialization**: Model is loaded once on server start.
- **Comprehensive Testing**: Both unit and integration tests provided.

### 7.2 Potential Improvements

- **GPU Deployment**: Modify `compute` and `gpu_count` in `cerebrium.toml`.
- **Better Logging**: Use `logging` module instead of `print`.
- **Async Inference**: Consider if async ONNX support becomes feasible.
- **Human-Readable Labels**: Map class ID to names.
- **CI/CD**: Add GitHub Actions for testing and deployment.
- **Model Registry**: Use MLflow or DVC for versioning.

---

## 8. Assignment Completion Status

 `convert_to_onnx.py`: Preprocessing + ONNX conversion  
 `model.py`: OnnxModel class  
 `test.py`: Unit tests  
 `cerebrium.toml`, `Dockerfile`: Deployment configs  
 `test_server.py`: Integration tests  
 `README`: Complete documentation

---

## 9. Loom Walkthrough Video

Walkthrough Video: [Model Deployment and Testing Overview on Cerebrium](https://www.loom.com/share/2f3f71594fdf4c2db5b05b8fa818f900)

The video demonstrates:

- Code walkthrough (`convert_to_onnx.py`, `model.py`, `main.py`)
- Deployment to Cerebrium
- API testing with `test_server.py`
- Project completion summary

