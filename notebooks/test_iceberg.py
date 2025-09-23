from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

def main():
    spark = SparkSession.builder \
        .appName("UpdateIcebergTable") \
        .master("spark://spark-iceberg:7077") \
        .getOrCreate()

    table_name = "db.city_age_avg"

    print(">>> Đọc bảng hiện có")
    df = spark.read.format("iceberg").load(table_name)
    df.show()

    # 1. Schema evolution có kiểm tra
    print(">>> Schema evolution: thêm cột cnt và updated_at nếu chưa có")
    cols = [f.name for f in spark.table(table_name).schema.fields]
    if "cnt" not in cols:
        spark.sql(f"ALTER TABLE {table_name} ADD COLUMN cnt bigint")
    if "updated_at" not in cols:
        spark.sql(f"ALTER TABLE {table_name} ADD COLUMN updated_at string")

    # 2. Aggregate lại: tính avg_age và count
    print(">>> Aggregate lại: tính avg_age và count")
    agg_df = df.groupBy("city") \
               .agg(F.avg("avg_age").alias("avg_age"),
                    F.count("*").alias("cnt")) \
               .withColumn("updated_at", F.lit(datetime.now().isoformat()))

    agg_df.show()

    # 3. Overwrite bảng với dữ liệu mới (schema đã khớp)
    agg_df.write.format("iceberg").mode("overwrite").save(table_name)
    print(">>> Đã overwrite dữ liệu mới vào bảng")

    # 4. Append thêm dữ liệu mới (ví dụ thành phố mới)
    new_data = [("Haiphong", 32.5, 1, datetime.now().isoformat())]
    new_cols = ["city", "avg_age", "cnt", "updated_at"]
    new_df = spark.createDataFrame(new_data, new_cols)

    new_df.write.format("iceberg").mode("append").save(table_name)
    print(">>> Đã append thêm dữ liệu mới vào bảng")

    # 5. Đọc lại toàn bộ bảng
    df_check = spark.read.format("iceberg").load(table_name)
    print(">>> Data sau khi update + append:")
    df_check.show()

    # 6. Kiểm tra snapshot history (ACID + time travel)
    print(">>> Snapshot history:")
    history = spark.sql(f"SELECT * FROM {table_name}.history")
    history.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
