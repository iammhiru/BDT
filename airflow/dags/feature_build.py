from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 9, 29),
    "retries": 0,
}

with DAG(
    dag_id="silver_to_feature",
    default_args=default_args,
    schedule_interval=None,  
    catchup=False,
    tags=["spark", "silver", "feature"],
) as dag:

    feature_task = SparkSubmitOperator(
        task_id="silver_to_feature_month",
        conn_id="spark_default",
        application="/opt/spark/jobs/build_feature.py",  
        name="silver-to-feature-month",
        application_args=[
            "--as_of_month", "202505"
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
