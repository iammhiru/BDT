from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder \
        .appName("ReadIcebergTable") \
        .master("spark://spark-iceberg:7077") \
        .getOrCreate()

    table_name = "db.city_age_avg"

    print(f">>> Đọc dữ liệu từ Iceberg table: {table_name}")
    df = spark.read.format("iceberg").load(table_name)

    print(">>> Schema:")
    df.printSchema()

    print(">>> Sample data:")
    df.show()

    print(">>> Snapshot history:")
    history = spark.sql(f"SELECT * FROM {table_name}.history")
    history.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
