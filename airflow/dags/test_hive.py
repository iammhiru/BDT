from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="create_hive_table_dag",
    default_args=default_args,
    description="DAG chạy PySpark để tạo bảng trong Hive Metastore",
    schedule_interval=None,  
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["spark", "hive", "minio"],
) as dag:

    create_hive_table = SparkSubmitOperator(
        task_id="create_hive_table",
        application="/opt/spark/jobs/create_hive_table.py",  
        conn_id="spark_default",  
        jars="/opt/extra-jars/hadoop-aws-3.3.2.jar,/opt/extra-jars/aws-java-sdk-bundle-1.11.901.jar",  
        conf={
            "spark.sql.catalogImplementation": "hive",
            "spark.hadoop.hive.metastore.uris": "thrift://hive-metastore:9083",
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
            "spark.hadoop.fs.s3a.access.key": "admin",
            "spark.hadoop.fs.s3a.secret.key": "password",
            "spark.hadoop.fs.s3a.path.style.access": "true",
        },
    )

    create_hive_table
