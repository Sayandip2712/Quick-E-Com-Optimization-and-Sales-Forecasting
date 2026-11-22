# Multi-stage / lightweight Python container for Quick E-Com Optimization Dashboard
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential for C-extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY . .

# Expose Plotly Dash default port
EXPOSE 8050

# Run with Gunicorn WSGI server
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "2"]
