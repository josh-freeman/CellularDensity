FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000


CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--access-logfile", "-", "--error-logfile", "-"]

