# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GROQ_API_KEY=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

WORKDIR /app

# Install system dependencies (for building some python packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Make the bench runner executable
RUN chmod +x /app/bench/run.sh

# The default command runs the mandatory benchmark evaluation script
CMD ["/app/bench/run.sh"]
