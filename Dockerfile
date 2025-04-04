FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY data/ /app/data/
COPY model/ /app/model/
COPY app/ /app/app/

# Train model and ensure files are kept
RUN mkdir -p /app/model && \
    python /app/model/train.py && \
    python /app/model/predict.py

# Explicitly copy results to final image
RUN cp /app/model/2022_predictions.csv /app/model/persisted_predictions.csv

CMD ["python", "/app/app/app.py"]
