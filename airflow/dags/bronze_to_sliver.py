from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 9, 29),
    "retries": 0,
}

with DAG(
    dag_id="bronze_to_silver",
    default_args=default_args,
    schedule_interval=None,  
    catchup=False,
    tags=["spark", "bronze", "silver"],
) as dag:

    snapshot_task = SparkSubmitOperator(
        task_id="bronze_to_silver_month",
        conn_id="spark_default",
        application="/opt/spark/jobs/bronze_to_silver.py",
        name="bronze-to-silver-month",
        application_args=[
            "--month", "202505" 
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

    daily_task = SparkSubmitOperator(
        task_id="bronze_to_silver_date",
        conn_id="spark_default",
        application="/opt/spark/jobs/bronze_to_silver.py",
        name="bronze-to-silver-date",
        application_args=[
            "--date", "20250501"  
        ],
        conf={
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
            "spark.hadoop.fs.s3a.access.key": "admin",
            "spark.hadoop.fs.s3a.secret.key": "password",
            "spark.hadoop.fs.s3a.path.style.access": "true",
        },
        jars="/opt/extra-jars/hadoop-aws-3.3.2.jar,/opt/extra-jars/aws-java-sdk-bundle-1.11.901.jar",
        verbose=True,
    )

    snapshot_task >> daily_task
