FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Set Gunicorn with increased workers and timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", \
     "--workers", "1", "--timeout", "120", \
     "--access-logfile", "-", "--error-logfile", "-"]

