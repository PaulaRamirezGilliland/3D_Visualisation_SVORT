# Use the official Python image from the Docker Hub with a specific Python version
FROM python:3.8.19-slim

# Set the working directory in the container
WORKDIR /app

# Install git (necessary for cloning the repository)
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Clone the GitHub repository
RUN git clone https://github.com/PaulaRamirezGilliland/3D_Visualisation_SVORT.git .

# Optionally, checkout a specific branch or commit (if needed)
# RUN git checkout your-branch-or-commit

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .
# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout=120

COPY . .

# Run your application or script
CMD ["/bin/bash"]

#CMD ["python", "main.py"]
