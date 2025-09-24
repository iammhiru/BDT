from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder \
        .appName("SparkMinIO_CSV") \
        .master("spark://spark-iceberg:7077") \
        .getOrCreate()

    data = [
        ("Alice", 25, "Hanoi"),
        ("Bob", 30, "Saigon"),
        ("Charlie", 35, "Danang"),
        ("David", 28, "Hue"),
    ]
    columns = ["name", "age", "city"]

    df = spark.createDataFrame(data, columns)

    print(">>> Data gốc:")
    df.show()

    csv_path = "s3a://my-bucket/rawdata/people_csv2/"
    df.write.mode("overwrite").option("header", "true").csv(csv_path)
    print(f">>> Đã ghi CSV vào MinIO tại {csv_path}")

    df_csv = spark.read.option("header", "true").csv(csv_path)
    print(">>> Data đọc lại từ MinIO (CSV):")
    df_csv.show()

    parquet_path = "s3a://my-bucket/parquetdata/people_parquet2/"
    df_csv.write.mode("overwrite").parquet(parquet_path)
    print(f">>> Đã ghi Parquet vào MinIO tại {parquet_path}")

    df_parquet = spark.read.parquet(parquet_path)
    print(">>> Data đọc lại từ MinIO (Parquet):")
    df_parquet.show()

    spark.stop()

if __name__ == "__main__":
    main()
