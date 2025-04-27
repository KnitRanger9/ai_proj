#!/bin/bash

# Build and start all services
docker-compose build
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Initialize Airflow
docker-compose exec airflow airflow db init
docker-compose exec airflow airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Initialize MLflow
docker-compose exec mlflow mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlflow-artifacts

# Initialize Prometheus
docker-compose exec prometheus prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.console.templates=/usr/share/prometheus/consoles

# Initialize Grafana
docker-compose exec grafana grafana-server \
    --config /etc/grafana/grafana.ini \
    --homepath /usr/share/grafana

# Initialize Alertmanager
docker-compose exec alertmanager alertmanager \
    --config.file=/etc/alertmanager/alertmanager.yml

echo "Deployment completed successfully!" 