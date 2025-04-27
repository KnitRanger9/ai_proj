from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))

from data_fetch import fetch_stock_data, save_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_data_pipeline',
    default_args=default_args,
    description='Pipeline for fetching and processing stock data',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

def fetch_data_task(symbol, **kwargs):
    """Task to fetch stock data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    df = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'))
    if df is not None:
        save_data(df, symbol, 'raw')

# Create tasks for each stock symbol
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
fetch_tasks = []

for symbol in symbols:
    fetch_task = PythonOperator(
        task_id=f'fetch_{symbol}',
        python_callable=fetch_data_task,
        op_kwargs={'symbol': symbol},
        dag=dag,
    )
    fetch_tasks.append(fetch_task)

# Create a task to process the data
process_task = BashOperator(
    task_id='process_data',
    bash_command='python /opt/airflow/dags/process_data.py',
    dag=dag,
)

# Set task dependencies
for task in fetch_tasks:
    task >> process_task 