from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 9, 29),
    "retries": 0,
}

with DAG(
    dag_id="check_silver_sample",
    default_args=default_args,
    schedule_interval=None,   # chạy thủ công
    catchup=False,
    tags=["silver", "check"],
) as dag:

    check_task = SparkSubmitOperator(
        task_id="check_silver_data",
        conn_id="spark_default",
        application="/opt/spark/jobs/check_silver_sample.py",
        name="check-silver-sample",
        verbose=True,
        jars="/opt/extra-jars/hadoop-aws-3.3.2.jar,/opt/extra-jars/aws-java-sdk-bundle-1.11.901.jar",
        conf={
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
            "spark.hadoop.fs.s3a.access.key": "admin",
            "spark.hadoop.fs.s3a.secret.key": "password",
            "spark.hadoop.fs.s3a.path.style.access": "true",
        },
    )
