FROM apache/airflow:2.9.0-python3.12

USER airflow

RUN pip install --no-cache-dir \
  "apache-airflow-providers-apache-spark" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.12.txt"

USER root
RUN apt update
RUN apt install -y default-jdk 
USER airflow
RUN pip install pyspark==3.5.5
RUN pip install minio