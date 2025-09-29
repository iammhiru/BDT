from airflow import DAG
from airflow.providers.trino.operators.trino import TrinoOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "retries": 0,
}

with DAG(
    dag_id="trino_create_table_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["trino", "hive", "warehouse"],
) as dag:

    # 1. Tạo schema (nếu chưa có)
    create_schema = TrinoOperator(
        task_id="create_schema",
        trino_conn_id="trino_default",
        sql="""
        CREATE SCHEMA IF NOT EXISTS hive.test_db
        WITH (location = 's3a://warehouse/test_db/')
        """,
    )

    # 2. Tạo bảng managed (KHÔNG external_location)
    create_table = TrinoOperator(
        task_id="create_table",
        trino_conn_id="trino_default",
        sql="""
        CREATE TABLE IF NOT EXISTS hive.test_db.people1 (
            id INT,
            name VARCHAR,
            age INT
        )
        WITH (
            format = 'PARQUET'
        )
        """,
    )

    # 3. Insert dữ liệu vào bảng
    insert_data = TrinoOperator(
        task_id="insert_data",
        trino_conn_id="trino_default",
        sql="""
        INSERT INTO hive.test_db.people1 (id, name, age)
        VALUES 
            (1, 'Alice', 30),
            (2, 'Bob', 25),
            (3, 'Charlie', 40)
        """,
    )

    # 4. Select dữ liệu để test
    select_data = TrinoOperator(
        task_id="select_data",
        trino_conn_id="trino_default",
        sql="SELECT * FROM hive.test_db.people1",
        handler=lambda rows: print(f"Query result from Trino: {rows}"),
    )

    create_schema >> create_table >> insert_data >> select_data
