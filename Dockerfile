FROM python:3.11-slim

# Install system dependencies


# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install Python packages
RUN pip install -r requirements.txt

# Run the app
CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8000"]
