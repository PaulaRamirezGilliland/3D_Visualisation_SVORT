# Use the official Python image from the Docker Hub with a specific Python version
FROM python:3.8.19-slim

# Set the working directory in the container
WORKDIR /app

# Install git and packages for running graphical applications
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    x11-apps  # For testing X11 applications


# Clone the GitHub repository into /app
RUN git clone https://github.com/PaulaRamirezGilliland/3D_Visualisation_SVORT.git /app

# Ensure the repository contents are correctly copied
RUN ls -R /app


# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout=120

# Set environment variable for X11 forwarding
ENV DISPLAY=:0

# Set the default command to run your Python script
CMD ["python", "3D_Visualisation_SVORT/main.py"]
