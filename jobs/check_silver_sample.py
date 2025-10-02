from pyspark.sql import SparkSession

def get_spark(app_name="check-silver", master="spark://spark-master:7077"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.warehouse.dir", "s3a://warehouse/hive/")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("hive.metastore.uris", "thrift://hive-metastore:9083") 
        .config("spark.sql.catalogImplementation", "hive") 
        .config("hive.metastore.warehouse.dir", "s3a://warehouse/hive/")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("hive.exec.dynamic.partition", "true")
        .config("hive.exec.dynamic.partition.mode", "nonstrict")
        .enableHiveSupport()
    )
    if master:
        spark = spark.master(master).getOrCreate()
    return spark

def check_table(spark, table_name, limit=10):
    print(f"\n===== Checking {table_name} =====")
    df = spark.table(table_name)
    df.printSchema()
    df.show(limit, truncate=False)

def main():
    spark = get_spark()

    # Check crm_silver
    check_table(spark, "silver.crm_silver", limit=5)

    # Check service_profile_silver
    check_table(spark, "silver.service_profile_silver", limit=5)

    spark.stop()

if __name__ == "__main__":
    main()
