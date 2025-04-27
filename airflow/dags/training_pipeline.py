from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add the ml directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ml'))

from training.train import train_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Pipeline for training stock prediction models',
    schedule_interval=timedelta(days=7),  # Train models weekly
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

def train_model_task(symbol, **kwargs):
    """Task to train a model for a specific stock."""
    data_path = f"/opt/airflow/data/processed/{symbol}_processed.csv"
    model_path = f"/opt/airflow/models/saved_models/{symbol}_model.h5"
    
    success = train_model(symbol, data_path, model_path)
    if not success:
        raise Exception(f"Failed to train model for {symbol}")

# Create tasks for each stock symbol
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
train_tasks = []

for symbol in symbols:
    train_task = PythonOperator(
        task_id=f'train_{symbol}',
        python_callable=train_model_task,
        op_kwargs={'symbol': symbol},
        dag=dag,
    )
    train_tasks.append(train_task)

# Create a task to evaluate models
evaluate_task = BashOperator(
    task_id='evaluate_models',
    bash_command='python /opt/airflow/dags/evaluate_models.py',
    dag=dag,
)

# Set task dependencies
for task in train_tasks:
    task >> evaluate_task 