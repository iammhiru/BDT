FROM apache/airflow:2.9.0-python3.12

USER airflow

RUN pip install --no-cache-dir \
  "apache-airflow-providers-apache-spark" \
  "apache-airflow-providers-trino" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.12.txt"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install pyspark==3.5.5 minio
