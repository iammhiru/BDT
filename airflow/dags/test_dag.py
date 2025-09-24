from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    "spark_csv_parquet_etl",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    spark_csv_to_parquet = SparkSubmitOperator(
        task_id="spark_csv_to_parquet",
        application="/opt/spark/jobs/test_csv_in_minio.py",  
        conn_id="spark_default",  
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
