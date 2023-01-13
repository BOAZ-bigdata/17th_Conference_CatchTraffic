import datetime as dt
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
# from airflow.providers.apache.spark.operators.spark_sql import SparkSqlOperator

import pandas as pd

# 인스타 크롤링 매주 1회
def crawlInstagramHashtagData():
    pass

# 크롤링한 데이터 전처리
def preprocessHashtagData():
    pass

# 해시태그 빈도수 분석
def countHashTagFrequency():
    pass

default_args = {
    'owner': 'eunhak',
    'start_date': dt.datetime(2023, 1, 16),
    'retries': 3,
    'retry_delay': dt.timedelta(minutes=5),
}

with DAG('hashtag_dag',
    default_args=default_args,
    schedule_interval=timedelta(weeks=1),
    ) as dag:

    # 시작
    print_starting = BashOperator(task_id='starting', bash_command='echo "start hashtag_dag"')

    # 인스타 크롤링 매주 1회
    crawl_hashtag = PythonOperator(task_id='crawlInstagramHashtag', python_callable=crawlInstagramHashtagData)

    # 크롤링한 인스타그램 해시태그 데이터 전처리
    preprocess_hashtagData = PythonOperator(task_id='preprocessHashtagData', python_callable=preprocessHashtagData)

    # 해시태그 빈도수 분석
    count_hashTag_frequency = PythonOperator(task_id='countHashTagFrequency', python_callable=countHashTagFrequency)

print_starting >> crawl_hashtag >> preprocess_hashtagData >> count_hashTag_frequency