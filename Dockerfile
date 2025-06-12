# Start from a stable Python base image. Using a slim version to keep the size down.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to prevent caching of pip packages and to ensure
# Python output is sent straight to the terminal without being buffered.
ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies.
# We use --no-cache-dir to reduce image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application directory, which contains our model and inference code,
# into the container's working directory.
COPY ./app /app

# The Cerebrium platform will automatically start the application