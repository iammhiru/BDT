import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, col

def get_spark(app_name, master="spark://spark-master:7077"):
    builder = (
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
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    return spark

def schema_crm():
    return StructType([
        StructField("customer_id", StringType(), False),
        StructField("account_id", StringType(), True),
        StructField("service_id", StringType(), True),
        StructField("full_name", StringType(), True),
        StructField("dob", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("household_composition", StringType(), True),
        StructField("household_member_age", StringType(), True),
        StructField("household_change_date", StringType(), True),
        StructField("address", StringType(), True),
        StructField("primary_msisdn", StringType(), True),
        StructField("msisdn_list", StringType(), True),
        StructField("district", StringType(), True),
        StructField("ward", StringType(), True),
        StructField("province", StringType(), True),
        StructField("housing_type", StringType(), True),
        StructField("corner_house_flag", StringType(), True),
        StructField("segment", StringType(), True),
        StructField("service_start_date", StringType(), True),
    ])


def schema_subscriptions():
    return StructType([
        StructField("subscription_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("service_id", StringType(), True),
        StructField("plan_id", StringType(), True),
        StructField("plan_name", StringType(), True),
        StructField("bandwidth_dl", StringType(), True),
        StructField("bandwidth_ul", StringType(), True),
        StructField("subscription_start_date", StringType(), True),
        StructField("subscription_end_date", StringType(), True),
        StructField("upgrade_event_flag", StringType(), True),
        StructField("upgrade_date", StringType(), True),
        StructField("address", StringType(), True),
        StructField("multi_site_flag", StringType(), True),
        StructField("multi_site_count", IntegerType(), True),
        StructField("address_change_flag", StringType(), True),
        StructField("address_change_date", StringType(), True),
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
        StructField("subscriber_id", StringType(), True),
        StructField("msisdn", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("cell_id", StringType(), True),
        StructField("cell_cluster_id", StringType(), True),
        StructField("location_area_code", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("duration", StringType(), True),
        StructField("imei", StringType(), True),
    ])

def clean_msisdn(col_expr):
    # B1: ép về string
    c = F.col(col_expr).cast("string")
    # B2: bỏ dấu ngoặc vuông, ngoặc kép, khoảng trắng
    c = F.regexp_replace(c, r'[\[\]\"]', "")
    c = F.trim(c)
    # B3: bỏ đuôi .0 nếu có
    c = F.regexp_replace(c, r'\.0$', "")
    # B4: nếu bắt đầu bằng 84 thì thay bằng 0
    c = F.regexp_replace(c, r'^\+?84', "0")
    c = F.when(F.col(col_expr).isNotNull(),
               F.regexp_replace(c, r'^84', "0")
              ).otherwise(None)
    return c

def clean_msisdn_list(col_expr):
    # Parse JSON string thành array
    arr = F.from_json(F.col(col_expr), ArrayType(StringType()))
    # Chuẩn hóa từng phần tử
    arr_clean = F.transform(
        arr,
        lambda x: F.regexp_replace(
            F.regexp_replace(F.regexp_replace(x, r'\.0$', ""), r'^\+?84', "0"),
            r'[\[\]\"]', ""
        )
    )
    return arr_clean


def create_silver_tables(spark):
    # Tạo database silver
    spark.sql("""
        CREATE DATABASE IF NOT EXISTS silver
        LOCATION 's3a://warehouse/hive/silver.db'
    """)

        # 1. CRM
    spark.sql("""
        CREATE TABLE IF NOT EXISTS silver.crm_silver (
            customer_id           STRING,
            account_id            STRING,
            service_id            STRING,
            full_name             STRING,
            dob                   DATE,
            age                   INT,
            gender                STRING,
            household_composition STRING,
            household_member_age  ARRAY<INT>,
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
            service_start_date    DATE,
            valid_from            DATE,
            valid_to              DATE,
            is_current            INT
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/silver.db/crm_silver'
    """)

    # 2. Subscriptions
    spark.sql("""
        CREATE TABLE IF NOT EXISTS silver.subscriptions_silver (
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
            address_change_date     DATE,
            valid_from              DATE,
            valid_to                DATE,
            is_current              INT
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/silver.db/subscriptions_silver'
    """)

    # 3. Service Profile
    spark.sql("""
        CREATE TABLE IF NOT EXISTS silver.service_profile_silver (
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
            wifi_clients_count_daily STRING,
            valid_from              DATE,
            valid_to                DATE,
            is_current              INT
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/silver.db/service_profile_silver'
    """)

    # 4. Usage Logs
    spark.sql("""
        CREATE TABLE IF NOT EXISTS silver.usage_silver (
            service_id     STRING,
            customer_id    STRING,
            timestamp      TIMESTAMP,
            date_stamp     DATE,
            hour           INT,
            uplink_bytes   BIGINT,
            downlink_bytes BIGINT,
            device_mac     STRING,
            device_id      STRING,
            session_id     STRING
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/silver.db/usage_silver'
    """)

    # 5. Tickets
    spark.sql("""
        CREATE TABLE IF NOT EXISTS silver.tickets_silver (
            ticket_id      STRING,
            customer_id    STRING,
            service_id     STRING,
            create_time    TIMESTAMP,
            close_time     TIMESTAMP,
            topic_group    STRING,
            keywords       STRING,
            description    STRING,
            site_visit_flag INT,
            resolved_flag   INT
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/silver.db/tickets_silver'
    """)

    # 6. Browsing Logs
    spark.sql("""
        CREATE TABLE IF NOT EXISTS silver.browsing_silver (
            customer_id  STRING,
            service_id   STRING,
            timestamp    TIMESTAMP,
            domain       STRING,
            topic        STRING,
            session_id   STRING,
            duration     DOUBLE
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/silver.db/browsing_silver'
    """)

    # 7. CDR / Mobility
    spark.sql("""
        CREATE TABLE IF NOT EXISTS silver.cdr_silver (
            cdr_id             STRING,
            msisdn             STRING,
            timestamp          TIMESTAMP,
            cell_id            STRING,
            cell_cluster_id    STRING,
            location_area_code STRING,
            event_type         STRING,
            duration           INT,
            imei               STRING
        )
        PARTITIONED BY (date STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/silver.db/cdr_silver'
    """)

def month_start_date(month_str):
    # month_str: 'YYYYMM' -> DATE(YYYY-MM-01)
    return F.to_date(F.concat_ws("", F.lit(month_str), F.lit("01")), "yyyyMMdd")

def prev_day_of(date_col):
    return F.date_sub(date_col, 1)

def table_cols(spark, table_name):
    return [f.name for f in spark.table(table_name).schema.fields]

def reorder_to_table(df, spark, table_name):
    # Chỉ reorder theo tên cột; giả định kiểu đã khớp
    cols = table_cols(spark, table_name)
    # Nếu thiếu cột nào, thêm NULL cast đúng kiểu
    target_schema = spark.table(table_name).schema
    existing = set(df.columns)
    select_exprs = []
    for f in target_schema.fields:
        if f.name in existing:
            select_exprs.append(F.col(f.name).cast(f.dataType).alias(f.name))
        else:
            select_exprs.append(F.lit(None).cast(f.dataType).alias(f.name))
    return df.select(*select_exprs)

def hash_row(df, cols):
    # hash ổn định cho so sánh thay đổi (cast về string để so sánh an toàn)
    return F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("§NULL§")) for c in cols]), 256)

def scd2_merge_dim(
    spark,
    df_new_typed,        
    silver_table,        
    natural_keys,        
    month_str            
):

    scd_cols = {"valid_from", "valid_to", "is_current", "month"}
    target_cols = table_cols(spark, silver_table)
    compare_cols = [c for c in target_cols if c not in set(natural_keys) | scd_cols]

    start_date = month_start_date(month_str)
    df_new = (
        df_new_typed
        .withColumn("valid_from", start_date.cast("date"))
        .withColumn("valid_to", F.lit(None).cast("date"))
        .withColumn("is_current", F.lit(1).cast("int"))
        .withColumn("month", F.lit(month_str))
    )

    # Đọc bản hiện hành (is_current=1) trên toàn bảng (không filter month)
    try:
        df_curr = spark.table(silver_table).where(F.col("is_current") == 1)
        has_curr = not df_curr.rdd.isEmpty()
    except Exception:
        has_curr = False

    if not has_curr:
        # Lần đầu: chỉ append new partition
        df_new_out = reorder_to_table(df_new, spark, silver_table)
        (df_new_out
            .write
            .mode("append")
            .insertInto(silver_table))
        return

    df_new_a = df_new.alias("n")
    df_curr_a = df_curr.alias("c")

    cond = [df_new_a[k] == df_curr_a[k] for k in natural_keys]

    joined = (
        df_new_a
        .join(
            df_curr_a.select(*natural_keys, "month", *compare_cols),
            cond,
            "left"
        )
    )

    new_hash = hash_row(df_new_a, [f"n.{c}" for c in compare_cols]).alias("new_hash")
    curr_hash = hash_row(df_curr_a, [f"c.{c}" for c in compare_cols]).alias("curr_hash")
    joined = joined.withColumn("new_hash", new_hash)
    # Chú ý: cần hash của df_curr -> join lại để lấy đúng curr_hash
    df_curr_h = df_curr.select(*natural_keys, hash_row(df_curr, compare_cols).alias("curr_hash"), F.col("month").alias("old_month"))

    j2 = df_new.join(df_curr_h, cond, "left")

    # Nhóm keys:
    # 1) brand new (không có curr)
    brand_new = j2.where(F.col("curr_hash").isNull()).select(*[df_new[k] for k in natural_keys]).dropDuplicates(natural_keys)

    # 2) changed (có curr nhưng hash khác)
    changed = j2.where((F.col("curr_hash").isNotNull()) & (F.col("curr_hash") != hash_row(df_new, compare_cols))) \
                .select(*[df_new[k] for k in natural_keys], F.col("old_month")).dropDuplicates()

    # 3) unchanged = còn lại (bỏ qua)

    # Expire các bản ghi cũ theo từng partition cũ
    expired_updates = None
    if not changed.rdd.isEmpty():
        affected_months = [r["old_month"] for r in changed.select("old_month").distinct().collect()]
        for m in affected_months:
            part_old = spark.table(silver_table).where(F.col("month") == m)
            keys_in_m = changed.where(F.col("old_month") == m).select(*natural_keys).dropDuplicates()
            # set is_current=0, valid_to = start_date - 1 day
            updated_part = (
                part_old.alias("p")
                .join(keys_in_m.alias("k"), [F.col(f"p.{k}") == F.col(f"k.{k}") for k in natural_keys], "left")
                .withColumn("is_current",
                            F.when(F.lit(True) & F.array(*(F.col(f"k.{k}") for k in natural_keys)).isNotNull(),
                                   F.lit(0)).otherwise(F.col("p.is_current")))
                .withColumn("valid_to",
                            F.when(F.array(*(F.col(f"k.{k}") for k in natural_keys)).isNotNull(),
                                   prev_day_of(start_date)).otherwise(F.col("p.valid_to")))
                .select("p.*")  # giữ nguyên thứ tự cột như bảng
            )
            expired_updates = updated_part if expired_updates is None else expired_updates.unionByName(updated_part)

    # Insert rows mới cần thêm (brand-new + changed)
    to_insert = df_new.join(brand_new, natural_keys, "left_semi") \
                      .unionByName(df_new.join(changed.select(*natural_keys).dropDuplicates(), natural_keys, "left_semi"))

    # Ghi: 1) overwrite các partition cũ bị ảnh hưởng  2) append các bản ghi mới tháng hiện tại
    if expired_updates is not None and not expired_updates.rdd.isEmpty():
        expired_out = reorder_to_table(expired_updates, spark, silver_table)
        (expired_out
            .write
            .mode("overwrite")              # chỉ overwrite những partition có trong DF nhờ dynamic
            .insertInto(silver_table))

    if not to_insert.rdd.isEmpty():
        to_insert_out = reorder_to_table(to_insert, spark, silver_table)
        (to_insert_out
            .write
            .mode("append")
            .insertInto(silver_table))

# --------------------------------------------------------------------------------------
# TRANSFORMS (cast về đúng kiểu Silver)
# --------------------------------------------------------------------------------------
def transform_crm(spark, month):
    df = spark.table("bronze.crm_bronze").where(F.col("month") == month)

    df_typed = (
        df.withColumn("dob", F.to_date("dob"))
          .withColumn("household_change_date", F.to_date("household_change_date"))
          .withColumn("service_start_date", F.to_date("service_start_date"))
          .withColumn("corner_house_flag", F.col("corner_house_flag").cast("int"))
          .withColumn("age", F.col("age").cast("int"))
          .withColumn("household_member_age", F.from_json("household_member_age", ArrayType(IntegerType())))
          .withColumn("primary_msisdn", clean_msisdn("primary_msisdn"))
          .withColumn("msisdn_list", F.col("msisdn_list")) 
    )

    scd2_merge_dim(
        spark=spark,
        df_new_typed=df_typed.select(
            "customer_id","account_id","service_id","full_name","dob","age","gender",
            "household_composition","household_member_age","household_change_date",
            "address","primary_msisdn","msisdn_list","district","ward","province",
            "housing_type","corner_house_flag","segment","service_start_date"
        ),
        silver_table="silver.crm_silver",
        natural_keys=["customer_id"],
        month_str=month
    )

def transform_subscriptions(spark, month):
    df = spark.table("bronze.subscriptions_bronze").where(F.col("month") == month)

    df_typed = (
        df.withColumn("bandwidth_dl", F.col("bandwidth_dl").cast("double"))
          .withColumn("bandwidth_ul", F.col("bandwidth_ul").cast("double"))
          .withColumn("subscription_start_date", F.to_date("subscription_start_date"))
          .withColumn("subscription_end_date", F.to_date("subscription_end_date"))
          .withColumn("upgrade_event_flag", F.col("upgrade_event_flag").cast("int"))
          .withColumn("upgrade_date", F.to_date("upgrade_date"))
          .withColumn("multi_site_flag", F.col("multi_site_flag").cast("int"))
          .withColumn("multi_site_count", F.col("multi_site_count").cast("int"))
          .withColumn("address_change_flag", F.col("address_change_flag").cast("int"))
          .withColumn("address_change_date", F.to_date("address_change_date"))
    )

    scd2_merge_dim(
        spark=spark,
        df_new_typed=df_typed.select(
            "subscription_id","customer_id","service_id","plan_id","plan_name",
            "bandwidth_dl","bandwidth_ul","subscription_start_date","subscription_end_date",
            "upgrade_event_flag","upgrade_date","address","multi_site_flag","multi_site_count",
            "address_change_flag","address_change_date"
        ),
        silver_table="silver.subscriptions_silver",
        natural_keys=["subscription_id"],
        month_str=month
    )

def transform_service_profile(spark, month):
    df = spark.table("bronze.service_profile_bronze").where(F.col("month") == month)

    df_typed = (
        df.withColumn("bandwidth_dl", F.col("bandwidth_dl").cast("double"))
          .withColumn("bandwidth_ul", F.col("bandwidth_ul").cast("double"))
          .withColumn("cpe_install_date", F.to_date("cpe_install_date"))
          .withColumn("cpe_replacement_date", F.to_date("cpe_replacement_date"))
          .withColumn("cpe_old_device_flag", F.col("cpe_old_device_flag").cast("int"))
          .withColumn("wifi_clients_count_daily", F.col("wifi_clients_count_daily").cast("string"))
    )

    scd2_merge_dim(
        spark=spark,
        df_new_typed=df_typed.select(
            "service_id","customer_id","plan_id","plan_name","bandwidth_dl","bandwidth_ul",
            "cpe_id","cpe_model","cpe_firmware_version","cpe_install_date","cpe_replacement_date",
            "cpe_old_device_flag","wifi_clients_count_daily"
        ),
        silver_table="silver.service_profile_silver",
        natural_keys=["service_id"],
        month_str=month
    )

def transform_usage(spark, date):
    df = spark.table("bronze.usage_bronze").where(F.col("date") == date)
    out = (
        df.withColumn("timestamp", F.to_timestamp("timestamp"))
          .withColumn("date_stamp", F.to_date("date_stamp"))
          .withColumn("hour", F.col("hour").cast("int"))
          .withColumn("uplink_bytes", F.col("uplink_bytes").cast("bigint"))
          .withColumn("downlink_bytes", F.col("downlink_bytes").cast("bigint"))
    )
    out = out.withColumn("date", F.lit(date))  # đảm bảo partition column tồn tại
    out = reorder_to_table(out, spark, "silver.usage_silver")
    (out.write.mode("overwrite").insertInto("silver.usage_silver"))

def transform_tickets(spark, date):
    df = spark.table("bronze.tickets_bronze").where(F.col("date") == date)
    out = (
        df.withColumn("create_time", F.to_timestamp("create_time"))
          .withColumn("close_time", F.to_timestamp("close_time"))
          .withColumn("site_visit_flag", F.col("site_visit_flag").cast("int"))
          .withColumn("resolved_flag", F.col("resolved_flag").cast("int"))
    )
    out = out.withColumn("date", F.lit(date))
    out = reorder_to_table(out, spark, "silver.tickets_silver")
    (out.write.mode("overwrite").insertInto("silver.tickets_silver"))

def transform_browsing(spark, date):
    df = spark.table("bronze.browsing_bronze").where(F.col("date") == date)
    out = (
        df.withColumn("timestamp", F.to_timestamp("timestamp"))
            .withColumn(
              "duration",
              F.when(F.col("duration").isNull(), F.lit(1.0))   # nếu null → 1.0
               .otherwise(F.col("duration").cast("double"))
            )
    )
    out = out.withColumn("date", F.lit(date))
    out = reorder_to_table(out, spark, "silver.browsing_silver")
    (out.write.mode("overwrite").insertInto("silver.browsing_silver"))

def transform_cdr(spark, date):
    df = spark.table("bronze.cdr_bronze").where(F.col("date") == date)
    out = (
        df.withColumn("timestamp", F.to_timestamp("timestamp"))
          .withColumn("duration", F.col("duration").cast("int"))
          .withColumn("msisdn", clean_msisdn("msisdn"))
          .withColumn("location_area_code", F.col("location_area_code").cast("string"))
    )
    out = out.withColumn("date", F.lit(date))
    out = reorder_to_table(out, spark, "silver.cdr_silver")
    (out.write.mode("overwrite").insertInto("silver.cdr_silver"))

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, help="YYYYMMDD")
    ap.add_argument("--month", type=str, help="YYYYMM")
    args = ap.parse_args()

    spark = get_spark("bronze-to-silver")
    create_silver_tables(spark)

    if args.month:
        transform_crm(spark, args.month)
        transform_subscriptions(spark, args.month)
        transform_service_profile(spark, args.month)

    if args.date:
        transform_usage(spark, args.date)
        transform_tickets(spark, args.date)
        transform_browsing(spark, args.date)
        transform_cdr(spark, args.date)

    spark.stop()

if __name__ == "__main__":
    main()