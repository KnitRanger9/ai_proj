#!/bin/bash

# Create necessary directories
mkdir -p backend/app/{api,core,models,schemas,services}
mkdir -p frontend/app/{static,templates}
mkdir -p ml/{models,data/{raw,processed,features},training,inference,utils}
mkdir -p airflow/{dags,logs,plugins}
mkdir -p mlflow/mlflow-artifacts
mkdir -p monitoring/{prometheus,grafana/dashboards,alertmanager}
mkdir -p tests/{integration,e2e,performance}
mkdir -p docs/{architecture,api,ml,user_guides}

# Create empty __init__.py files
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/core/__init__.py
touch backend/app/models/__init__.py
touch backend/app/schemas/__init__.py
touch backend/app/services/__init__.py
touch frontend/app/__init__.py

# Create .gitkeep files to preserve empty directories
find . -type d -empty -exec touch {}/.gitkeep \;

# Set up Python virtual environments
python -m venv backend/venv
python -m venv frontend/venv
python -m venv ml/venv

# Install dependencies
cd backend && source venv/bin/activate && pip install -r requirements.txt
cd ../frontend && source venv/bin/activate && pip install -r requirements.txt
cd ../ml && source venv/bin/activate && pip install -r requirements.txt

echo "Setup completed successfully!" 