FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix paths and remove redundant RUN
RUN mkdir -p /app/data /app/model && \
    python /app/model/train.py && \
    python /app/model/predict.py

CMD ["python", "/app/app/app.py"]
