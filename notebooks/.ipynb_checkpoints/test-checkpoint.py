from pyspark.sql import SparkSession

def main():
    # Tạo Spark session (master có thể đổi thành local[*] nếu bạn chỉ muốn test local)
    spark = SparkSession.builder \
        .appName("SparkMinIO_CSV2Parquet") \
        .master("spark://spark-iceberg:7077") \
        .getOrCreate()

    # Đọc CSV (giả sử bạn mount file CSV vào container Spark, ví dụ /home/iceberg/notebooks/data/input.csv)
    input_csv = "/home/iceberg/notebooks/data/input.csv"
    df = spark.read.option("header", "true").csv(input_csv)

    print(">>> Schema CSV:")
    df.printSchema()
    print(">>> Một vài dòng CSV:")
    df.show(5)

    # Ghi DataFrame xuống MinIO dạng Parquet
    output_path = "s3a://my-bucket/test_parquet/"
    df.write.mode("overwrite").parquet(output_path)
    print(f">>> Đã ghi DataFrame CSV -> Parquet vào MinIO tại {output_path}")

    # Đọc lại từ MinIO
    df2 = spark.read.parquet(output_path)
    print(">>> Data đọc lại từ MinIO:")
    df2.show(5)

    spark.stop()

if __name__ == "__main__":
    main()
