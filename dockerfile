# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    python3 \
    python3-pip \
    cmake \
    clang \
    lld

# Install Ollama using the official script
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy the Streamlit app code
COPY app.py .

# Expose the Streamlit port
EXPOSE 8501

# Start Ollama, pull the DeepSeek R1 model, and run the Streamlit app
CMD bash -c "ollama serve & sleep 10 && ollama pull deepseek-r1:1.5b && streamlit run app.py --server.address=0.0.0.0"