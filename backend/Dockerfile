# Use a lightweight Python version
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install System Dependencies (Tesseract & Database tools)
# We force 'tesseract-ocr' installation here
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the requirements file first (for better caching)
COPY requirements.txt .

# 3. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]