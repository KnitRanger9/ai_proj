FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 