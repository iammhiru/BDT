from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    "spark_trino_etl",
    default_args=default_args,
    description="Run Spark job on MinIO data and validate with Trino",
    schedule_interval=None,   
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["lakehouse", "spark", "trino"],
) as dag:

    spark_csv_job = SparkSubmitOperator(
        task_id="spark_csv_job",
        application="/opt/spark/jobs/test_csv_to_iceberg.py", 
        jars="/opt/extra-jars/hadoop-aws-3.3.2.jar,/opt/extra-jars/aws-java-sdk-bundle-1.11.901.jar,/opt/extra-jars/iceberg-spark-runtime-3.5_2.12-1.10.0.jar",
        conf={
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
        "spark.hadoop.fs.s3a.access.key": "admin",
        "spark.hadoop.fs.s3a.secret.key": "password",
        "spark.hadoop.fs.s3a.path.style.access": "true"
        },
        conn_id="spark-air", 
        verbose=True,
    )