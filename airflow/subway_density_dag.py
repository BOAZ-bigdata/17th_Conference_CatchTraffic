import datetime as dt
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
# from airflow.providers.apache.spark.operators.spark_sql import SparkSqlOperator

import pandas as pd

# 지하철 혼잡도 api 크롤링
def crawlSubwayData():
    pass

# 크롤링한 데이터 전처리
def preprocessSubwayData():
    pass

# 지하철 혼잡도 예측
def predictSubwayDensity():
    pass

default_args = {
    'owner': 'eunhak',
    'start_date': dt.datetime(2023, 1, 16),
    'retries': 3,
    'retry_delay': dt.timedelta(minutes=5),
}

with DAG('subway_density_dag',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    ) as dag:

    # 시작
    print_starting = BashOperator(task_id='starting', bash_command='echo "start subway_density_dag"')

    # 지하철 혼잡도 api 크롤링
    crawl_subway_data = PythonOperator(task_id='crawlSubwayDensity', python_callable=crawlSubwayData)

    # 크롤링한 지하철 데이터 전처리
    preprocess_subwayData = PythonOperator(task_id='preprocessSubwayData', python_callable=preprocessSubwayData)

    # 지하철 혼잡도 예측
    predict_subway_density = PythonOperator(task_id='predictSubwayDensity', python_callable=predictSubwayDensity)

print_starting >> crawl_subway_data >> preprocess_subwayData >> predict_subway_density