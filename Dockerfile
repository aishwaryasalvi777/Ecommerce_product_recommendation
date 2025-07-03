# Base Python image
FROM python:3.9-slim

# 👇 Install libgomp and other utilities
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your code
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Start both apps
CMD ["./start.sh"]
