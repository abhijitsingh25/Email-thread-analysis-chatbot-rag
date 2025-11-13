# ------------ BASIC DOCKERFILE (NO GPU, NO CUDA) --------------

FROM python:3.10-slim

# Set working directory
# COPY . /app/


WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose backend + UI ports
EXPOSE 8000
EXPOSE 8501

# Start backend + Streamlit together
CMD bash -c "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & \
             streamlit run ui/frontend.py --server.port=8501 --server.address=0.0.0.0"
