FROM python:3.11-slim

WORKDIR /app

# System packages needed to compile some Python wheels (scipy, scikit-learn)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source files + dataset
COPY . .

# Pre-train the ML model at build time so container starts instantly
# (If the dataset is large you can remove this and let it train on first request)
RUN python ml_predictor.py

# Uvicorn bound to 0.0.0.0 so it's reachable from other containers and the host
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
