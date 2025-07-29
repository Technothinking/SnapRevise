FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y tesseract-ocr libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm


COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
