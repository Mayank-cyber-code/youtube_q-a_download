# Use official Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Make vectorstores directory for FAISS indexes
RUN mkdir -p vectorstores

# Expose port your app listens on
EXPOSE 10000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
