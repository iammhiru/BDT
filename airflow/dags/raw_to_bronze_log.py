from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 6, 1),  
    "end_date": datetime(2025, 8, 31),   
    "retries": 0,
}

with DAG(
    dag_id="raw_to_bronze_daily",
    default_args=default_args,
    schedule_interval="@daily", 
    catchup=True,
) as dag:

    daily_task = SparkSubmitOperator(
        task_id="raw_to_bronze_date",
        conn_id="spark_default",
        application="/opt/spark/jobs/raw_to_bronze.py",
        name="raw-to-bronze-date",
        application_args=[
            "--date", "{{ execution_date.strftime('%Y%m%d') }}"
        ],
        jars="/opt/extra-jars/hadoop-aws-3.3.2.jar,/opt/extra-jars/aws-java-sdk-bundle-1.11.901.jar",
        conf={
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
            "spark.hadoop.fs.s3a.access.key": "admin",
            "spark.hadoop.fs.s3a.secret.key": "password",
            "spark.hadoop.fs.s3a.path.style.access": "true",
        },
        verbose=True,
    )
