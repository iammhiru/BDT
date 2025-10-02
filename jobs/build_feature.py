#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build feature tables (Gold-Feature) from Silver layer for a given as_of_month.

Run:
  spark-submit --master spark://spark-master:7077 \
    /opt/spark/jobs/build_features.py --as_of_month 2025-05
"""

import argparse
from pyspark.sql import SparkSession, functions as F, Window as W
from pyspark.sql.types import DoubleType, IntegerType

# ---------------------------
# Spark bootstrap (Hive external)
# ---------------------------
def spark_session(app="silver_to_feature"):
    return (
        SparkSession.builder
        .appName(app)
        .master("spark://spark-master:7077")
        .config("spark.sql.warehouse.dir", "s3a://warehouse/hive/")
        .config("hive.metastore.uris", "thrift://hive-metastore:9083")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.shuffle.partitions", "200")
        .enableHiveSupport()
        .getOrCreate()
    )

# ---------------------------
# Time helpers
# ---------------------------
def month_bounds(as_of_month: str):
    """Return Spark SQL expressions for first and last day of as_of_month."""
    s_first = F.to_date(F.lit(f"{as_of_month}-01"))
    s_last  = F.last_day(s_first)
    return s_first, s_last

def win_days(s_last, days: int):
    """Return [start, end] date window inclusive: [S-(days-1), S]."""
    return F.date_sub(s_last, days - 1), s_last

# ---------------------------
# Infra: ensure feature DB & tables
# ---------------------------
def ensure_feature_db_and_tables(spark):
    spark.sql("""
        CREATE DATABASE IF NOT EXISTS feature
        LOCATION 's3a://warehouse/hive/feature.db'
    """)

    # 1) Usage 30d
    spark.sql("""
        CREATE TABLE IF NOT EXISTS feature.f_usage_30d (
            service_id STRING,
            customer_id STRING,
            as_of_month STRING,
            calc_window INT,
            night_small_ul_streaksum_30d INT,
            small_ul_all_day_streaksum_30d INT,
            home_absence_days_from_usage_30d INT,
            calc_ts TIMESTAMP
        )
        PARTITIONED BY (as_of_month)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/feature.db/f_usage_30d'
    """)

    # 2) Tickets 90d + intent 30d
    spark.sql("""
        CREATE TABLE IF NOT EXISTS feature.f_tickets_90d (
            service_id STRING,
            customer_id STRING,
            as_of_month STRING,
            calc_window INT,
            camera_install_inquiry_30d_flag INT,
            camera_security_ticket_cnt_90d INT,
            wifi_coverage_ticket_cnt_90d INT,
            calc_ts TIMESTAMP
        )
        PARTITIONED BY (as_of_month)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/feature.db/f_tickets_90d'
    """)

    # 3) Browsing 30d
    spark.sql("""
        CREATE TABLE IF NOT EXISTS feature.f_browsing_30d (
            service_id STRING,
            customer_id STRING,
            as_of_month STRING,
            calc_window INT,
            browsing_camera_topic_hits_30d INT,
            calc_ts TIMESTAMP
        )
        PARTITIONED BY (as_of_month)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/feature.db/f_browsing_30d'
    """)

    # 4) Subscriptions events (90d)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS feature.f_subs_90d (
            service_id STRING,
            customer_id STRING,
            as_of_month STRING,
            calc_window INT,
            service_address_change_90d_flag INT,
            broadband_upgrade_90d_flag INT,
            calc_ts TIMESTAMP
        )
        PARTITIONED BY (as_of_month)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/feature.db/f_subs_90d'
    """)

    # 5) Service profile (180d)
    spark.sql("""
        CREATE TABLE IF NOT EXISTS feature.f_profile_180d (
            service_id STRING,
            customer_id STRING,
            as_of_month STRING,
            calc_window INT,
            cpe_replacement_180d_flag INT,
            cpe_old_device_flag INT,
            avg_daily_wifi_clients_30d DOUBLE,
            calc_ts TIMESTAMP
        )
        PARTITIONED BY (as_of_month)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/feature.db/f_profile_180d'
    """)

    # 6) CDR 30d
    spark.sql("""
        CREATE TABLE IF NOT EXISTS feature.f_cdr_30d (
            service_id STRING,
            customer_id STRING,
            as_of_month STRING,
            calc_window INT,
            away_days_from_cdr_30d INT,
            multi_area_mobility_days_30d INT,
            early_out_late_in_flag_30d INT,
            calc_ts TIMESTAMP
        )
        PARTITIONED BY (as_of_month)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/feature.db/f_cdr_30d'
    """)

# ---------------------------
# Compute helpers
# ---------------------------
def _consecutive_streaksum(df, id_col, date_col, flag_col):
    """
    Sum of lengths of consecutive-day streaks where flag_col==1.
    Uses "days_since_epoch - row_number" trick to group runs.
    """
    # Keep only "ok" days
    ok = df.where(F.col(flag_col) == 1)
    if ok.rdd.isEmpty():
        return df.select(id_col).distinct().withColumn("streaksum", F.lit(0))

    # days since epoch
    ok = ok.withColumn("day_num", F.datediff(F.col(date_col), F.lit("1970-01-01")))
    w = W.partitionBy(id_col).orderBy(F.col(date_col).asc())
    ok = ok.withColumn("rn", F.row_number().over(w))
    ok = ok.withColumn("grp", F.col("day_num") - F.col("rn"))
    runs = ok.groupBy(id_col, "grp").agg(F.count(F.lit(1)).alias("run_len"))
    agg  = runs.groupBy(id_col).agg(F.sum("run_len").alias("streaksum"))
    return agg

# ---------------------------
# Feature builders
# ---------------------------
def build_usage_30d(spark, as_of_month):
    s_first, s_last = month_bounds(as_of_month)
    start30, end30  = win_days(s_last, 30)

    # base usage (UTC)
    usage = (spark.table("silver.usage_silver")
             .withColumn("event_date", F.to_date(F.col("date")))  # partition is string in silver
             .where((F.col("event_date") >= start30) & (F.col("event_date") <= end30))
             .select("service_id", "customer_id", "event_date", "hour", "uplink_bytes", "downlink_bytes"))

    usage = usage.withColumn("total_bytes", F.col("uplink_bytes") + F.col("downlink_bytes"))

    # Night small UL: 0..5h, 5MB..60MB
    L = F.lit(5 * 1024 * 1024)
    U = F.lit(60 * 1024 * 1024)
    BULK = F.lit(500 * 1024 * 1024)

    night = usage.where((F.col("hour") >= 0) & (F.col("hour") <= 5)) \
                 .withColumn("hour_hit", F.when((F.col("uplink_bytes") >= L) & (F.col("uplink_bytes") <= U), 1).otherwise(0)) \
                 .withColumn("has_bulk", F.when(F.col("uplink_bytes") > BULK, 1).otherwise(0))

    nightly = (night.groupBy("service_id", "customer_id", "event_date")
                    .agg(F.sum("hour_hit").alias("hit_cnt"),
                         F.max("has_bulk").alias("bulk")))
    nightly = nightly.withColumn("night_ok", F.when((F.col("hit_cnt") >= 4) & (F.col("bulk") == 0), 1).otherwise(0))

    # Streaksum for nights
    ns = _consecutive_streaksum(nightly, "service_id", "event_date", "night_ok") \
         .withColumnRenamed("streaksum", "night_small_ul_streaksum_30d")

    # All-day small UL: >=16/24 hours in small range (simplified)
    allday = usage.withColumn("hour_hit", F.when((F.col("uplink_bytes") >= L) & (F.col("uplink_bytes") <= U), 1).otherwise(0)) \
                  .groupBy("service_id", "customer_id", "event_date") \
                  .agg(F.sum("hour_hit").alias("h24"))
    allday = allday.withColumn("day_ok", F.when(F.col("h24") >= 16, 1).otherwise(0))
    ds = _consecutive_streaksum(allday, "service_id", "event_date", "day_ok") \
         .withColumnRenamed("streaksum", "small_ul_all_day_streaksum_30d")

    # Absence: total_bytes < 200MB
    daily = usage.groupBy("service_id", "customer_id", "event_date").agg(F.sum("total_bytes").alias("dbytes"))
    absence = (daily.groupBy("service_id", "customer_id")
               .agg(F.sum(F.when(F.col("dbytes") < F.lit(200 * 1024 * 1024), 1).otherwise(0)).alias("home_absence_days_from_usage_30d")))

    # assemble
    base = (usage.select("service_id", "customer_id").dropDuplicates())
    out = (base.join(ns, "service_id", "left")
                .join(ds, "service_id", "left")
                .join(absence, ["service_id", "customer_id"], "left")
                .na.fill(0)
                .withColumn("as_of_month", F.lit(as_of_month))
                .withColumn("calc_window", F.lit(30))
                .withColumn("calc_ts", F.current_timestamp())
           )

    # write
    spark.sql(f"ALTER TABLE feature.f_usage_30d DROP IF EXISTS PARTITION (as_of_month='{as_of_month}')")
    (out.repartition(1)
        .write.mode("append")
        .insertInto("feature.f_usage_30d"))

def build_tickets_90d(spark, as_of_month):
    s_first, s_last = month_bounds(as_of_month)
    start30, end30  = win_days(s_last, 30)
    start90, end90  = win_days(s_last, 90)

    t = (spark.table("silver.tickets_silver")
         .withColumn("event_date", F.to_date(F.col("date")))
         .where((F.col("event_date") >= start90) & (F.col("event_date") <= end90))
         .select("service_id", "customer_id", "create_time", "event_date", "topic_group", "keywords"))

    # Intent 30d: camera install inquiry
    t30 = t.where((F.col("event_date") >= start30) & (F.col("event_date") <= end30))
    cam_flag = (t30.where(
                    F.lower(F.coalesce(F.col("topic_group"), F.lit(""))).rlike("camera|security|cctv") |
                    F.lower(F.coalesce(F.col("keywords"), F.lit(""))).rlike("lắp|lap|khảo|chuông|báo động|alarm|cctv|camera")
                )
                .select("service_id").distinct()
                .withColumn("camera_install_inquiry_30d_flag", F.lit(1)))

    # Camera/security ticket count 90d (gộp thô theo day+topic)
    cam90 = (t.where(F.lower(F.coalesce(F.col("topic_group"), F.lit(""))).rlike("camera|security|nat|ddns|storage|motion|wiring"))
             .withColumn("gday", F.to_date(F.col("create_time")))
             .groupBy("service_id", "gday", "topic_group").count()
             .groupBy("service_id").agg(F.count("*").alias("camera_security_ticket_cnt_90d")))

    # WiFi coverage 90d
    wifi90 = (t.where(F.lower(F.coalesce(F.col("topic_group"), F.lit(""))).rlike("wifi|weak|coverage|router"))
              .withColumn("gday", F.to_date(F.col("create_time")))
              .groupBy("service_id", "gday", "topic_group").count()
              .groupBy("service_id").agg(F.count("*").alias("wifi_coverage_ticket_cnt_90d")))

    base = t.select("service_id", "customer_id").dropDuplicates()
    out = (base.join(cam_flag, "service_id", "left")
                .join(cam90, "service_id", "left")
                .join(wifi90, "service_id", "left")
                .na.fill({"camera_install_inquiry_30d_flag": 0,
                          "camera_security_ticket_cnt_90d": 0,
                          "wifi_coverage_ticket_cnt_90d": 0})
                .withColumn("as_of_month", F.lit(as_of_month))
                .withColumn("calc_window", F.lit(90))
                .withColumn("calc_ts", F.current_timestamp()))

    spark.sql(f"ALTER TABLE feature.f_tickets_90d DROP IF EXISTS PARTITION (as_of_month='{as_of_month}')")
    (out.repartition(1)
        .write.mode("append")
        .insertInto("feature.f_tickets_90d"))

def build_browsing_30d(spark, as_of_month):
    s_first, s_last = month_bounds(as_of_month)
    start30, end30  = win_days(s_last, 30)

    b = (spark.table("silver.browsing_silver")
         .withColumn("event_date", F.to_date(F.col("date")))
         .where((F.col("event_date") >= start30) & (F.col("event_date") <= end30))
         .select("service_id", "customer_id", "topic", "domain"))

    brows = (b.where(F.lower(F.coalesce(F.col("topic"), F.lit(""))).rlike("camera|cctv|alarm|chuông|báo động"))
               .groupBy("service_id").agg(F.count("*").alias("browsing_camera_topic_hits_30d")))

    base = b.select("service_id", "customer_id").dropDuplicates()
    out = (base.join(brows, "service_id", "left")
                .na.fill({"browsing_camera_topic_hits_30d": 0})
                .withColumn("as_of_month", F.lit(as_of_month))
                .withColumn("calc_window", F.lit(30))
                .withColumn("calc_ts", F.current_timestamp()))

    spark.sql(f"ALTER TABLE feature.f_browsing_30d DROP IF EXISTS PARTITION (as_of_month='{as_of_month}')")
    (out.repartition(1)
        .write.mode("append")
        .insertInto("feature.f_browsing_30d"))

def build_subs_90d(spark, as_of_month):
    s_first, s_last = month_bounds(as_of_month)
    start90, end90  = win_days(s_last, 90)

    subs = (spark.table("silver.subscriptions_silver")
            .where(F.col("month") == as_of_month)
            .select("service_id", "customer_id",
                    "address_change_flag", "address_change_date",
                    "upgrade_event_flag", "upgrade_date"))

    out = (subs
           .withColumn("service_address_change_90d_flag",
                       F.when((F.col("address_change_flag") == 1) &
                              (F.col("address_change_date").between(start90, end90)), 1).otherwise(0))
           .withColumn("broadband_upgrade_90d_flag",
                       F.when((F.col("upgrade_event_flag") == 1) &
                              (F.col("upgrade_date").between(start90, end90)), 1).otherwise(0))
           .select("service_id", "customer_id",
                   "service_address_change_90d_flag",
                   "broadband_upgrade_90d_flag")
           .withColumn("as_of_month", F.lit(as_of_month))
           .withColumn("calc_window", F.lit(90))
           .withColumn("calc_ts", F.current_timestamp()))

    spark.sql(f"ALTER TABLE feature.f_subs_90d DROP IF EXISTS PARTITION (as_of_month='{as_of_month}')")
    (out.repartition(1)
        .write.mode("append")
        .insertInto("feature.f_subs_90d"))

def build_profile_180d(spark, as_of_month):
    s_first, s_last = month_bounds(as_of_month)
    start180, end180 = win_days(s_last, 180)

    prof = (spark.table("silver.service_profile_silver")
            .where(F.col("month") == as_of_month)
            .select("service_id", "customer_id",
                    "cpe_replacement_date", "cpe_old_device_flag",
                    "wifi_clients_count_daily"))

    out = (prof
           .withColumn("cpe_replacement_180d_flag",
                       F.when(F.col("cpe_replacement_date").between(start180, end180), 1).otherwise(0))
           # avg_daily_wifi_clients_30d: để None nếu chuỗi chưa parse được (đủ dùng V1)
           .withColumn("avg_daily_wifi_clients_30d", F.lit(None).cast(DoubleType()))
           .select("service_id", "customer_id",
                   "cpe_replacement_180d_flag",
                   "cpe_old_device_flag",
                   "avg_daily_wifi_clients_30d")
           .withColumn("as_of_month", F.lit(as_of_month))
           .withColumn("calc_window", F.lit(180))
           .withColumn("calc_ts", F.current_timestamp()))

    spark.sql(f"ALTER TABLE feature.f_profile_180d DROP IF EXISTS PARTITION (as_of_month='{as_of_month}')")
    (out.repartition(1)
        .write.mode("append")
        .insertInto("feature.f_profile_180d"))

def build_cdr_30d(spark, as_of_month):
    s_first, s_last = month_bounds(as_of_month)
    start30, end30  = win_days(s_last, 30)

    # Map service_id -> customer_id (từ CRM tháng đó)
    crm = (spark.table("silver.crm_silver")
           .where((F.col("month") == as_of_month) & (F.col("is_current") == 1))
           .select("service_id", "customer_id", "primary_msisdn"))

    # CDR 30d
    cdr = (spark.table("silver.cdr_silver")
           .withColumn("event_date", F.to_date(F.col("date")))
           .where((F.col("event_date") >= start30) & (F.col("event_date") <= end30))
           .select("msisdn", "timestamp", "event_date", "cell_cluster_id"))

    # V1: join qua primary_msisdn (đơn giản & đủ nhanh)
    cdrj = (cdr.join(crm, cdr.msisdn == crm.primary_msisdn, "inner")
               .select("service_id", "customer_id", "event_date", "cell_cluster_id"))

    # Xác định "home cluster" = mode của cluster trong 30 ngày
    w = W.partitionBy("service_id").orderBy(F.desc("cnt"))
    home = (cdrj.groupBy("service_id", "cell_cluster_id").count().withColumnRenamed("count", "cnt")
                .withColumn("rk", F.row_number().over(w))
                .where(F.col("rk") == 1)
                .select("service_id", F.col("cell_cluster_id").alias("home_cluster_id")))

    c = (cdrj.join(home, "service_id", "left")
            .withColumn("away_flag", F.when(F.col("cell_cluster_id") != F.col("home_cluster_id"), 1).otherwise(0)))

    # away_days: nếu trong ngày có away_flag=1
    away = (c.groupBy("service_id", "customer_id", "event_date")
             .agg(F.max("away_flag").alias("is_away"))
             .groupBy("service_id", "customer_id")
             .agg(F.sum("is_away").alias("away_days_from_cdr_30d")))

    # multi_area_mobility_days: >=2 cluster khác home trong ngày (đơn giản)
    multi_area = (c.groupBy("service_id", "customer_id", "event_date")
                    .agg(F.countDistinct(F.when(F.col("cell_cluster_id") != F.col("home_cluster_id"),
                                               F.col("cell_cluster_id"))).alias("nonhome_cnt"))
                    .groupBy("service_id", "customer_id")
                    .agg(F.sum(F.when(F.col("nonhome_cnt") >= 2, 1).otherwise(0)).alias("multi_area_mobility_days_30d")))

    # early_out_late_in_flag_30d (rất đơn giản): nếu có >=10 ngày nonhome về đêm → 1
    eoli = (away.withColumn("flag_day", F.when(F.col("away_days_from_cdr_30d") >= 10, 1).otherwise(0))
                 .select("service_id", "customer_id", "flag_day")
                 .groupBy("service_id", "customer_id")
                 .agg(F.max("flag_day").alias("early_out_late_in_flag_30d")))

    base = cdrj.select("service_id", "customer_id").dropDuplicates()
    out = (base.join(away, ["service_id", "customer_id"], "left")
                .join(multi_area, ["service_id", "customer_id"], "left")
                .join(eoli, ["service_id", "customer_id"], "left")
                .na.fill({"away_days_from_cdr_30d": 0,
                          "multi_area_mobility_days_30d": 0,
                          "early_out_late_in_flag_30d": 0})
                .withColumn("as_of_month", F.lit(as_of_month))
                .withColumn("calc_window", F.lit(30))
                .withColumn("calc_ts", F.current_timestamp()))

    spark.sql(f"ALTER TABLE feature.f_cdr_30d DROP IF EXISTS PARTITION (as_of_month='{as_of_month}')")
    (out.repartition(1)
        .write.mode("append")
        .insertInto("feature.f_cdr_30d"))

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as_of_month", required=True, help="YYYY-MM, e.g. 2025-05")
    args = ap.parse_args()

    spark = spark_session()
    ensure_feature_db_and_tables(spark)

    # Build features per group (independent tables)
    build_usage_30d(spark, args.as_of_month)
    build_tickets_90d(spark, args.as_of_month)
    build_browsing_30d(spark, args.as_of_month)
    build_subs_90d(spark, args.as_of_month)
    build_profile_180d(spark, args.as_of_month)
    build_cdr_30d(spark, args.as_of_month)

    print(f"[OK] Built feature tables for as_of_month={args.as_of_month}")

if __name__ == "__main__":
    main()
