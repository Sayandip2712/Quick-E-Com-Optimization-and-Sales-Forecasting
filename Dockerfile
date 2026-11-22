# Multi-stage / lightweight Python container for Quick E-Com Optimization Dashboard
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build-essential / gcc / openmp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application source code and data assets
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Run with Gunicorn WSGI server on port 7860
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "2"]
