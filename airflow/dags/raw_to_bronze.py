from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 9, 29),
    "retries": 0,
}

with DAG(
    dag_id="raw_to_bronze",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    # Job xử lý snapshot hàng tháng (CRM, subscriptions, service_profile)
    snapshot_task = SparkSubmitOperator(
        task_id="raw_to_bronze_month",
        conn_id="spark_default",
        application="/opt/spark/jobs/raw_to_bronze.py",
        name="raw-to-bronze-month",
        application_args=[
            "--month", "202505"   # lấy YYYYMM từ execution_date
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

    # Job xử lý dữ liệu hằng ngày (usage, tickets, browsing, cdr)
    daily_task = SparkSubmitOperator(
        task_id="raw_to_bronze_date",
        conn_id="spark_default",
        application="/opt/spark/jobs/raw_to_bronze.py",
        name="raw-to-bronze-date",
        application_args=[
            "--date", "20250501"   # lấy YYYYMMDD từ execution_date
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
