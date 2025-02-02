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
    lld \
    netcat \
    libffi-dev \
    openssh-client

# Install Ollama using the official script
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add wait-for-it script
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

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

# Health check for Ollama server
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD nc -z localhost 11434 || exit 1

# Start services
CMD bash -c "/wait-for-it.sh localhost:11434 --timeout=60 --strict -- \
    ollama pull deepseek-r1 && \
    streamlit run app.py --server.address=0.0.0.0 & \
    ollama serve"