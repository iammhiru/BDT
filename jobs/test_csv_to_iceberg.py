from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    spark = SparkSession.builder \
        .appName("CSVtoIceberg") \
        .master("spark://spark-iceberg:7077") \
        .getOrCreate()

    input_path = "s3a://my-bucket/rawdata/people_csv/"
    df = spark.read.option("header", "true").csv(input_path)
    df = df.withColumn("age", df["age"].cast("int"))

    agg_df = df.groupBy("city").agg(F.avg("age").alias("avg_age"))

    spark.sql("CREATE NAMESPACE IF NOT EXISTS db")

    spark.sql("""
        CREATE TABLE IF NOT EXISTS db.city_age_avg (
            city string,
            avg_age double
        )
        USING iceberg
        PARTITIONED BY (city)
    """)

    agg_df.write.format("iceberg").mode("overwrite").save("db.city_age_avg")

    print(">>> Đã ghi dữ liệu vào Iceberg table: db.city_age_avg")

    df_check = spark.read.format("iceberg").load("db.city_age_avg")
    df_check.show()

    history = spark.sql("SELECT * FROM db.city_age_avg.history")
    history.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
