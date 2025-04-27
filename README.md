# Stock Prediction Application

A comprehensive stock prediction system that combines machine learning, data engineering, and web technologies to provide accurate stock price predictions.

## Architecture

The application consists of several microservices:
- FastAPI Backend: Handles API requests and business logic
- Flask Frontend: Provides the user interface
- ML Service: Manages model training and inference
- Airflow: Orchestrates data pipelines and model training
- MLflow: Tracks experiments and model versions
- Monitoring: Prometheus and Grafana for system monitoring

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Git
- DVC (for data version control)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-prediction-app.git
cd stock-prediction-app
```

2. Set up the environment:
```bash
./scripts/setup.sh
```

3. Start the services:
```bash
docker-compose up -d
```

4. Access the application:
- Frontend: http://localhost:5000
- Backend API: http://localhost:8000
- MLflow: http://localhost:5000
- Grafana: http://localhost:3000

## Development

### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
flask run
```

### ML Development
```bash
cd ml
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Documentation

- [Architecture Overview](docs/architecture/README.md)
- [API Documentation](docs/api/README.md)
- [ML Model Documentation](docs/ml/README.md)
- [User Guides](docs/user_guides/README.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 