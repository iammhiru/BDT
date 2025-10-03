import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F

def get_spark(app_name, master=None):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.warehouse.dir", "s3a://warehouse/hive/")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("hive.metastore.uris", "thrift://hive-metastore:9083") 
        .config("spark.sql.catalogImplementation", "hive") 
        .config("hive.metastore.warehouse.dir", "s3a://warehouse/hive/")
        .enableHiveSupport()
    )
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    return spark

def normalize_msisdn_list(df):
    """ Chuẩn hóa msisdn_list từ raw CSV """
    return (
        df.withColumn(
                "msisdn_list",
                F.regexp_replace("msisdn_list", r'^\[|\]$', "")  # bỏ [] ngoài
            ).withColumn(
                "msisdn_list",
                F.split(F.regexp_replace("msisdn_list", r'[^0-9,]', ""), ",")  # thành array
            ).withColumn(
                "msisdn_list",
                F.expr("filter(msisdn_list, x -> x != '')")  # bỏ rỗng
            ).withColumn(
                "msisdn_list",
                F.expr("transform(msisdn_list, x -> regexp_replace(x, '^84', '0'))")  # chuẩn hóa số
            ).withColumn(
                "msisdn_list",
                F.to_json("msisdn_list")  # stringify lại để khớp schema STRING
            )
    )


def schema_crm():
    return StructType([
        StructField("customer_id", StringType(), False),
        StructField("account_id", StringType(), True),
        StructField("service_id", StringType(), True),
        StructField("full_name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("household_composition", StringType(), True),
        StructField("household_member_age", StringType(), True),
        StructField("household_change_date", DateType(), True),
        StructField("address", StringType(), True),
        StructField("primary_msisdn", StringType(), True),
        StructField("msisdn_list", StringType(), True),
        StructField("district", StringType(), True),
        StructField("ward", StringType(), True),
        StructField("province", StringType(), True),
        StructField("housing_type", StringType(), True),
        StructField("corner_house_flag", IntegerType(), True),
        StructField("segment", StringType(), True),
        StructField("service_start_date", DateType(), True),
    ])


def schema_subscriptions():
    return StructType([
        StructField("subscription_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("service_id", StringType(), True),
        StructField("plan_id", StringType(), True),
        StructField("plan_name", StringType(), True),
        StructField("bandwidth_dl", DoubleType(), True),
        StructField("bandwidth_ul", DoubleType(), True),
        StructField("subscription_start_date", DateType(), True),
        StructField("subscription_end_date", DateType(), True),
        StructField("upgrade_event_flag", IntegerType(), True),
        StructField("upgrade_date", DateType(), True),
        StructField("address", StringType(), True),
        StructField("multi_site_flag", IntegerType(), True),
        StructField("multi_site_count", IntegerType(), True),
        StructField("address_change_flag", IntegerType(), True),
        StructField("address_change_date", DateType(), True),
    ])


def schema_service_profile():
    return StructType([
        StructField("service_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("plan_id", StringType(), True),
        StructField("plan_name", StringType(), True),
        StructField("bandwidth_dl", DoubleType(), True),
        StructField("bandwidth_ul", DoubleType(), True),
        StructField("cpe_id", StringType(), True),
        StructField("cpe_model", StringType(), True),
        StructField("cpe_firmware_version", StringType(), True),
        StructField("cpe_install_date", DateType(), True),
        StructField("cpe_replacement_date", DateType(), True),
        StructField("cpe_old_device_flag", IntegerType(), True),
        StructField("wifi_clients_count_daily", StringType(), True),
    ])


def schema_usage():
    return StructType([
        StructField("service_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("date_stamp", StringType(), True),
        StructField("hour", StringType(), True),
        StructField("uplink_bytes", LongType(), True),
        StructField("downlink_bytes", LongType(), True),
        StructField("device_mac", StringType(), True),
        StructField("device_id", StringType(), True),
        StructField("session_id", StringType(), True),
    ])


def schema_tickets():
    return StructType([
        StructField("ticket_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("service_id", StringType(), True),
        StructField("create_time", StringType(), True),
        StructField("close_time", StringType(), True),
        StructField("topic_group", StringType(), True),
        StructField("keywords", StringType(), True),
        StructField("description", StringType(), True),
        StructField("site_visit_flag", StringType(), True),
        StructField("resolved_flag", StringType(), True),
    ])


def schema_browsing():
    return StructType([
        StructField("customer_id", StringType(), True),
        StructField("service_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("domain", StringType(), True),
        StructField("topic", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("duration", StringType(), True),
    ])


def schema_cdr():
    return StructType([
        StructField("cdr_id", StringType(), False),
        StructField("msisdn", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("cell_id", StringType(), True),
        StructField("cell_cluster_id", StringType(), True),
        StructField("location_area_code", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("duration", StringType(), True),
        StructField("imei", StringType(), True),
    ])

def create_bronze_tables(spark):
    # Tạo database bronze
    spark.sql("""
        CREATE DATABASE IF NOT EXISTS bronze
        LOCATION 's3a://warehouse/hive/bronze.db'
    """)

    # 1. CRM (snapshot theo month)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS bronze.crm_bronze (
            customer_id           STRING,
            account_id            STRING,
            service_id            STRING,
            full_name             STRING,
            dob                   DATE,
            age                   INT,
            gender                STRING,
            household_composition STRING,
            household_member_age  STRING,
            household_change_date DATE,
            address               STRING,
            primary_msisdn        STRING,
            msisdn_list           STRING,
            district              STRING,
            ward                  STRING,
            province              STRING,
            housing_type          STRING,
            corner_house_flag     INT,
            segment               STRING,
            service_start_date    DATE
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/bronze.db/crm_bronze'
    """)

    # 2. Subscriptions (snapshot theo month)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS bronze.subscriptions_bronze (
            subscription_id        STRING,
            customer_id            STRING,
            service_id             STRING,
            plan_id                STRING,
            plan_name              STRING,
            bandwidth_dl           DOUBLE,
            bandwidth_ul           DOUBLE,
            subscription_start_date DATE,
            subscription_end_date   DATE,
            upgrade_event_flag      INT,
            upgrade_date            DATE,
            address                 STRING,
            multi_site_flag         INT,
            multi_site_count        INT,
            address_change_flag     INT,
            address_change_date     DATE
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/bronze.db/subscriptions_bronze'
    """)

    # 3. Service Profile (snapshot theo month)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS bronze.service_profile_bronze (
            service_id             STRING,
            customer_id            STRING,
            plan_id                STRING,
            plan_name              STRING,
            bandwidth_dl           DOUBLE,
            bandwidth_ul           DOUBLE,
            cpe_id                 STRING,
            cpe_model              STRING,
            cpe_firmware_version   STRING,
            cpe_install_date       DATE,
            cpe_replacement_date   DATE,
            cpe_old_device_flag    INT,
            wifi_clients_count_daily STRING
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/bronze.db/service_profile_bronze'
    """)

    # 4. Usage Logs (dữ liệu hằng ngày)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS bronze.usage_bronze (
            service_id     STRING,
            customer_id    STRING,
            timestamp      STRING,
            date_stamp     STRING,
            hour           STRING,
            uplink_bytes   BIGINT,
            downlink_bytes BIGINT,
            device_mac     STRING,
            device_id      STRING,
            session_id     STRING
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/bronze.db/usage_bronze'
    """)

    # 5. Tickets (dữ liệu hằng ngày)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS bronze.tickets_bronze (
            ticket_id      STRING,
            customer_id    STRING,
            service_id     STRING,
            create_time    STRING,
            close_time     STRING,
            topic_group    STRING,
            keywords       STRING,
            description    STRING,
            site_visit_flag STRING,
            resolved_flag   STRING
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/bronze.db/tickets_bronze'
    """)

    # 6. Browsing Logs (dữ liệu hằng ngày)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS bronze.browsing_bronze (
            customer_id  STRING,
            service_id   STRING,
            timestamp    STRING,
            domain       STRING,
            topic        STRING,
            session_id   STRING,
            duration     STRING
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/bronze.db/browsing_bronze'
    """)

    # 7. CDR / Mobility (dữ liệu hằng ngày)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS bronze.cdr_bronze (
            cdr_id             STRING,
            msisdn             STRING,
            timestamp          STRING,
            cell_id            STRING,
            cell_cluster_id    STRING,
            location_area_code STRING,
            event_type         STRING,
            duration           STRING,
            imei               STRING
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/bronze.db/cdr_bronze'
    """)



from pyspark.sql import functions as F

def process_snapshot(spark, source, schema, month):
    """
    Xử lý dữ liệu snapshot theo month (CRM, Subscriptions, Service Profile)
    """
    input_path = f"s3a://raw-stage/{source}/month={month}/*.csv"
    output_table = f"bronze.{source}_bronze"

    df = (
        spark.read.option("header", True)
        .option("quote", "\"")       # FIX HERE: tránh lỗi parse CSV có dấu , hoặc "
        .option("escape", "\"")      # FIX HERE
        .schema(schema)
        .csv(input_path)
        .withColumn("month", F.lit(month))
    )

    # Xử lý đặc biệt theo source
    if source == "crm":
        df = normalize_msisdn_list(df)   # FIX HERE: chuẩn hóa msisdn_list

    if source == "service_profile":
        df = df.withColumn("wifi_clients_count_daily", F.col("wifi_clients_count_daily").cast("string"))

    # Ghi dữ liệu vào bảng bronze đã tạo
    (df.write
        .mode("overwrite")
        .format("parquet")
        .insertInto(output_table, overwrite=True))


    print(f"[{source}] Snapshot {month} → bronze done.")


def process_daily(spark, source, schema, date):
    """
    Xử lý dữ liệu log theo date (Usage, Tickets, Browsing, CDR)
    """
    if source == "usage":
        input_path = f"s3a://raw-stage/{source}/usage_raw_10min_hour/date={date}/*.csv"
    else:
        input_path = f"s3a://raw-stage/{source}/{source}_raw/date={date}/*.csv"
    output_table = f"bronze.{source}_bronze"

    # Đọc dữ liệu thô
    df = (
        spark.read.option("header", True)
        .schema(schema)
        .csv(input_path)
        .withColumn("date", F.lit(date))
    )

    # Ghi dữ liệu vào bảng bronze đã tạo
    (df.write
       .mode("overwrite")
       .format("parquet")
       .insertInto(output_table, overwrite=True))

    print(f"[{source}] Daily {date} → bronze done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Ngày dữ liệu dạng YYYYMMDD")
    parser.add_argument("--month", type=str, help="Tháng dữ liệu dạng YYYYMM")
    args = parser.parse_args()

    spark = get_spark("raw-to-bronze", master="spark://spark-master:7077")

    create_bronze_tables(spark)

    if args.month:
        process_snapshot(spark, "crm", schema_crm(), args.month)
        process_snapshot(spark, "subscriptions", schema_subscriptions(), args.month)
        process_snapshot(spark, "service_profile", schema_service_profile(), args.month)

    if args.date:
        process_daily(spark, "usage", schema_usage(), args.date)
        process_daily(spark, "tickets", schema_tickets(), args.date)
        process_daily(spark, "browsing", schema_browsing(), args.date)
        process_daily(spark, "cdr", schema_cdr(), args.date)

    spark.stop()


if __name__ == "__main__":
    main()





