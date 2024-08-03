# Use the official Python image from the Docker Hub with a specific Python version
FROM python:3.8.19-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout=120

# Copy the rest of the application code into the container at /app
COPY . .

# Default command (can be overridden at runtime)
CMD ["python", "main.py"]
