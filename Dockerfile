FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only raw data and code (no model files)
COPY data/ /app/data/
COPY model/train.py /app/model/
COPY model/predict.py /app/model/
COPY model/model_utils.py /app/model/
COPY app/ /app/app/

# Build-time training (no cache)
RUN mkdir -p /app/model && \
    python /app/model/train.py && \
    python /app/model/predict.py

CMD ["python", "/app/app/app.py"]
