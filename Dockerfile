FROM python:3.11-slim

# System dependencies (minimized)
RUN apt-get update && apt-get install -y \
    gcc libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8000"]
