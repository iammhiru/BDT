from pyspark.sql import SparkSession

def main():
    spark = (
        SparkSession.builder
        .appName("CreateHiveTable")
        .master("spark://spark-master:7077") 
        .enableHiveSupport()
        .getOrCreate()
    )

    input_path = "s3a://my-bucket/rawdata/people_csv2/"

    # Tạo DataFrame
    df = spark.read.option("header", "true").csv(input_path)

    print("=== Sample Data ===")
    df.show(5)

    # Tên bảng trong Hive Metastore (schema = default)
    table_name = "default.people"

    # Ghi DataFrame vào Hive (Parquet, overwrite)
    (
        df.write
        .mode("overwrite")
        .format("parquet")
        .option("path", "s3a://warehouse/hive/people/")   # nơi thực lưu data
        .saveAsTable(table_name)                          # đăng ký vào Hive Metastore
    )

    print(f"Table {table_name} has been created and registered in Hive Metastore.")

    # Check bảng vừa tạo
    spark.sql(f"SHOW TABLES IN default").show()
    spark.sql(f"SELECT * FROM {table_name} LIMIT 5").show()

    spark.stop()

if __name__ == "__main__":
    main()
