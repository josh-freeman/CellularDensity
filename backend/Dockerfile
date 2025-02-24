FROM python:3.12.4

# Set working directory
WORKDIR /app

# Copy and install backend dependencies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose backend port
EXPOSE 8080

# Command to run the backend application
CMD ["python", "app.py"]
