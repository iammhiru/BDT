import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta

def get_spark(app_name, master="spark://spark-master:7077"):
    """Khởi tạo Spark Session"""
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.warehouse.dir", "s3a://warehouse/hive/")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("hive.metastore.uris", "thrift://hive-metastore:9083") 
        .config("spark.sql.catalogImplementation", "hive") 
        .config("hive.metastore.warehouse.dir", "s3a://warehouse/hive/")
        .config("spark.sql.adaptive.enabled", "true")
        .enableHiveSupport()
    )
    if master:
        builder = builder.master(master)
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    return spark

def get_previous_month(current_month):
    """Tính tháng trước - FIX cho data hiện tại"""
    return current_month  # Dùng chính tháng đó

def clean_msisdn(col_expr):
    """Hàm làm sạch msisdn"""
    c = F.col(col_expr).cast("string")
    c = F.regexp_replace(c, r'[\[\]\"]', "")
    c = F.trim(c)
    c = F.regexp_replace(c, r'\.0$', "")
    c = F.regexp_replace(c, r'^\+?84', "0")
    return c

def get_msisdn_to_customer_mapping(spark, data_month):
    """Tạo mapping từ msisdn sang customer_id"""
    df_crm = spark.table("silver.crm_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1) \
        .select("customer_id", "primary_msisdn", "msisdn_list")
    
    # Clean primary_msisdn
    df_primary = df_crm.select(
        "customer_id",
        clean_msisdn("primary_msisdn").alias("msisdn")
    ).filter(F.col("msisdn").isNotNull())
    
    # Xử lý msisdn_list
    df_list = df_crm.filter(F.col("msisdn_list").isNotNull()) \
        .withColumn("msisdn_array", 
                   F.split(F.regexp_replace(F.regexp_replace(F.col("msisdn_list"), r'[\[\]\"]', ''), r'\s+', ''), ',')) \
        .select("customer_id", F.explode("msisdn_array").alias("msisdn_raw")) \
        .withColumn("msisdn", clean_msisdn("msisdn_raw")) \
        .filter(F.col("msisdn").isNotNull()) \
        .select("customer_id", "msisdn")
    
    df_mapping = df_primary.unionByName(df_list).distinct()
    return df_mapping

def get_customer_base(spark, data_month):
    """Lấy danh sách customer base từ CRM"""
    return spark.table("silver.crm_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1) \
        .select("customer_id").distinct()

def create_feature_tables(spark):
    """Tạo feature tables với các metric đã rút gọn"""
    spark.sql("CREATE DATABASE IF NOT EXISTS gold LOCATION 's3a://warehouse/hive/gold.db'")
    
    spark.sql("""
        CREATE TABLE IF NOT EXISTS gold.customer_features_monthly (
            customer_id STRING,
            snapshot_month STRING,
            
            -- NHÓM 1: Ý ĐỊNH/QUAN TÂM TRỰC TIẾP
            camera_install_inquiry_30d_flag INT,
            camera_security_ticket_cnt_90d INT,
            browsing_camera_topic_hits_30d INT,
            
            -- NHÓM 2: HÀNH VI SỬ DỤNG TẠI NHÀ
            night_small_ul_streaksum_30d INT,
            small_ul_all_day_streaksum_30d INT,
            
            -- NHÓM 3: BỐI CẢNH ĐỊA CHỈ
            housing_type_street_house_flag INT,
            corner_house_probability_score DOUBLE,
            
            -- NHÓM 4: SỰ KIỆN DỊCH VỤ
            service_address_change_90d_flag INT,
            broadband_upgrade_90d_flag INT,
            
            -- NHÓM 5: HIỆN DIỆN/VẮNG NHÀ
            home_absence_days_from_usage_30d INT,
            away_days_from_cdr_30d INT,
            multi_area_mobility_days_30d INT,
            
            -- NHÓM 6: HỖ TRỢ KỸ THUẬT
            wifi_coverage_ticket_cnt_90d INT,
            security_issue_ticket_cnt_180d INT,
            
            -- NHÓM 7: KINH DOANH
            retail_peak_usage_pattern_90d_flag INT,
            pos_related_ticket_cnt_180d INT,
            multi_site_service_count INT,
            avg_daily_wifi_clients_30d DOUBLE,
            
            -- NHÓM 8: THỜI GIAN NHẠY CẢM
            post_movein_period_flag INT,
            neighbor_security_incident_flag INT,
            
            -- NHÓM 9: NHÂN KHẨU HỌC
            household_has_children_flag INT,
            household_has_elderly_flag INT,
            household_composition_change_90d_flag INT,
            
            updated_at TIMESTAMP,
            processed_date DATE
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION 's3a://warehouse/hive/gold.db/customer_features_monthly'
    """)

def calculate_direct_intent_features(spark, snapshot_month, data_month):
    """NHÓM 1: Ý ĐỊNH/QUAN TÂM TRỰC TIẾP"""
    print("🔍 Calculating direct intent features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_30d = F.date_sub(snapshot_date, 29)
    start_date_90d = F.date_sub(snapshot_date, 89)
    
    df_customers = get_customer_base(spark, data_month)
    df_tickets = spark.table("silver.tickets_silver")
    df_browsing = spark.table("silver.browsing_silver")
    
    # 1. camera_install_inquiry_30d_flag
    camera_keywords = ["camera", "cctv", "lắp đặt", "khảo sát vị trí", "chuông cửa", "báo động"]
    keyword_pattern = "|".join(camera_keywords)
    
    df_camera_inquiry = (
        df_tickets
        .filter((F.col("create_time") >= start_date_30d) & (F.col("create_time") <= snapshot_date))
        .filter(
            F.lower(F.col("topic_group")).rlike(keyword_pattern) |
            F.lower(F.col("keywords")).rlike(keyword_pattern) |
            F.lower(F.col("description")).rlike(keyword_pattern)
        )
        .select("customer_id").distinct()
        .withColumn("camera_install_inquiry_30d_flag", F.lit(1))
    )
    
    # 2. camera_security_ticket_cnt_90d - với deduplication
    security_topics = ["camera", "security", "nat_ddns", "video_storage", "motion_alert", "wiring_outdoor"]
    
    # Deduplicate tickets trong 48h cho cùng service_id và topic_group
    ticket_window = Window.partitionBy("service_id", "topic_group").orderBy("create_time")
    df_dedup_tickets = (
        df_tickets
        .filter((F.col("create_time") >= start_date_90d) & (F.col("create_time") <= snapshot_date))
        .filter(F.lower(F.col("topic_group")).isin([t.lower() for t in security_topics]))
        .withColumn("prev_ticket_time", F.lag("create_time").over(ticket_window))
        .withColumn("hours_since_prev", 
                   F.when(F.col("prev_ticket_time").isNotNull(), 
                         (F.unix_timestamp("create_time") - F.unix_timestamp("prev_ticket_time")) / 3600)
                    .otherwise(None))
        .withColumn("is_duplicate", 
                   F.when((F.col("hours_since_prev").isNotNull()) & (F.col("hours_since_prev") <= 48), 1)
                    .otherwise(0))
        .filter(F.col("is_duplicate") == 0)
    )
    
    df_camera_tickets = (
        df_dedup_tickets
        .groupBy("customer_id")
        .agg(F.count("ticket_id").alias("camera_security_ticket_cnt_90d"))
    )
    
    # 3. browsing_camera_topic_hits_30d
    df_browsing_hits = (
        df_browsing
        .filter((F.col("timestamp") >= start_date_30d) & (F.col("timestamp") <= snapshot_date))
        .filter(
            F.lower(F.col("topic")).rlike(keyword_pattern) |
            F.lower(F.col("domain")).rlike(keyword_pattern)
        )
        .groupBy("customer_id")
        .agg(F.count("timestamp").alias("browsing_camera_topic_hits_30d"))
    )
    
    df_result = (
        df_customers
        .join(df_camera_inquiry, "customer_id", "left")
        .join(df_camera_tickets, "customer_id", "left")
        .join(df_browsing_hits, "customer_id", "left")
        .fillna(0)
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_result

def calculate_usage_pattern_features(spark, snapshot_month, data_month):
    """NHÓM 2: HÀNH VI SỬ DỤNG TẠI NHÀ"""
    print("🔍 Calculating usage pattern features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_30d = F.date_sub(snapshot_date, 29)
    
    df_customers = get_customer_base(spark, data_month)
    df_usage = spark.table("silver.usage_silver")
    
    # Lấy date range cho partition filter
    start_date_py = datetime.strptime(data_month + '01', '%Y%m%d').date()
    date_range = [(start_date_py - timedelta(days=x)).strftime('%Y%m%d') for x in range(30, 0, -1)]
    
    df_usage_30d = (
        df_usage
        .filter(F.col("date").isin(date_range))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("uplink_mb", F.col("uplink_bytes") / (1024 * 1024))
        .withColumn("downlink_mb", F.col("downlink_bytes") / (1024 * 1024))
    )
    
    # 4. night_small_ul_streaksum_30d - Chuỗi đêm có UL nhỏ-đều (0-5h)
    df_night_hourly = (
        df_usage_30d
        .filter(F.col("hour").between(0, 5))
        .groupBy("customer_id", "date_stamp", "hour")
        .agg(F.sum("uplink_mb").alias("hourly_ul_mb"))
        .withColumn("hour_achieved", 
                   F.when((F.col("hourly_ul_mb") >= 5) & (F.col("hourly_ul_mb") <= 60), 1).otherwise(0))
    )
    
    # Tìm các đêm đạt (≥4/6 giờ đạt, không có bulk UL >500MB/h)
    df_night_daily = (
        df_night_hourly
        .groupBy("customer_id", "date_stamp")
        .agg(
            F.sum("hour_achieved").alias("achieved_hours"),
            F.max("hourly_ul_mb").alias("max_hourly_ul")
        )
        .withColumn("night_achieved", 
                   F.when((F.col("achieved_hours") >= 4) & (F.col("max_hourly_ul") <= 500), 1).otherwise(0))
    )
    
    # Tìm chuỗi liên tiếp ≥3 đêm
    window_spec = Window.partitionBy("customer_id").orderBy("date_stamp")
    df_night_streaks = (
        df_night_daily
        .filter(F.col("night_achieved") == 1)
        .withColumn("prev_date", F.lag("date_stamp").over(window_spec))
        .withColumn("gap", F.datediff("date_stamp", "prev_date"))
        .withColumn("streak_id", 
                   F.sum(F.when((F.col("prev_date").isNull()) | (F.col("gap") > 1), 1).otherwise(0))
                   .over(window_spec))
        .groupBy("customer_id", "streak_id")
        .agg(F.count("*").alias("streak_length"))
        .filter(F.col("streak_length") >= 3)
        .groupBy("customer_id")
        .agg(F.sum("streak_length").alias("night_small_ul_streaksum_30d"))
    )
    
    # 5. small_ul_all_day_streaksum_30d - Chuỗi ngày có UL nhỏ-đều cả ngày
    df_all_day_hourly = (
        df_usage_30d
        .groupBy("customer_id", "date_stamp", "hour")
        .agg(F.sum("uplink_mb").alias("hourly_ul_mb"))
        .withColumn("hour_achieved", 
                   F.when((F.col("hourly_ul_mb") >= 5) & (F.col("hourly_ul_mb") <= 60), 1).otherwise(0))
    )
    
    # Tính gap giữa các giờ không đạt
    df_hour_gaps = (
        df_all_day_hourly
        .withColumn("prev_hour_achieved", F.lag("hour_achieved").over(
            Window.partitionBy("customer_id", "date_stamp").orderBy("hour")
        ))
        .withColumn("gap_start", 
                   F.when((F.col("hour_achieved") == 0) & 
                          (F.col("prev_hour_achieved") == 1), 1)
                   .otherwise(0))
        .withColumn("gap_id", 
                   F.sum("gap_start").over(
                       Window.partitionBy("customer_id", "date_stamp").orderBy("hour")
                   ))
    )
    
    df_gap_lengths = (
        df_hour_gaps
        .filter(F.col("hour_achieved") == 0)
        .groupBy("customer_id", "date_stamp", "gap_id")
        .agg(F.count("*").alias("gap_length"))
        .groupBy("customer_id", "date_stamp")
        .agg(F.max("gap_length").alias("max_gap_length"))
    )
    
    df_all_day_daily = (
        df_all_day_hourly
        .groupBy("customer_id", "date_stamp")
        .agg(
            F.sum("hour_achieved").alias("achieved_hours"),
            F.count("hour").alias("total_hours")
        )
        .join(df_gap_lengths, ["customer_id", "date_stamp"], "left")
        .fillna(0)
        # Ngày đạt: ≥16/24 giờ đạt, mỗi khung có ≥3 giờ đạt, không có gap >3 giờ
        .withColumn("day_achieved", 
                   F.when((F.col("achieved_hours") >= 16) & 
                          (F.col("total_hours") >= 20) &
                          (F.col("max_gap_length") <= 3), 1)
                   .otherwise(0))
    )
    
    df_day_streaks = (
        df_all_day_daily
        .filter(F.col("day_achieved") == 1)
        .withColumn("prev_date", F.lag("date_stamp").over(window_spec))
        .withColumn("gap", F.datediff("date_stamp", "prev_date"))
        .withColumn("streak_id", 
                   F.sum(F.when((F.col("prev_date").isNull()) | (F.col("gap") > 1), 1).otherwise(0))
                   .over(window_spec))
        .groupBy("customer_id", "streak_id")
        .agg(F.count("*").alias("streak_length"))
        .filter(F.col("streak_length") >= 3)
        .groupBy("customer_id")
        .agg(F.sum("streak_length").alias("small_ul_all_day_streaksum_30d"))
    )
    
    df_result = (
        df_customers
        .join(df_night_streaks, "customer_id", "left")
        .join(df_day_streaks, "customer_id", "left")
        .fillna(0)
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_result

def calculate_housing_context_features(spark, snapshot_month, data_month):
    """NHÓM 3: BỐI CẢNH ĐỊA CHỈ"""
    print("🔍 Calculating housing context features...")
    
    df_crm = spark.table("silver.crm_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1)
    
    # 6. housing_type_street_house_flag
    # 7. corner_house_probability_score
    df_housing = (
        df_crm
        .select("customer_id", "housing_type", "corner_house_flag")
        .distinct()
        .withColumn("housing_type_street_house_flag",
                   F.when(F.lower(F.col("housing_type")).isin(["nhà phố", "biệt thự", "mặt tiền", "nhà góc"]), 1)
                    .otherwise(0))
        .withColumn("corner_house_probability_score",
                   F.when(F.col("corner_house_flag") == 1, 0.9)
                    .when(F.lower(F.col("housing_type")).contains("góc"), 0.7)
                    .when(F.lower(F.col("housing_type")).contains("mặt tiền"), 0.5)
                    .otherwise(0.1))
        .select("customer_id", "housing_type_street_house_flag", "corner_house_probability_score")
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_housing

def calculate_service_event_features(spark, snapshot_month, data_month):
    """NHÓM 4: SỰ KIỆN DỊCH VỤ"""
    print("🔍 Calculating service event features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_90d = F.date_sub(snapshot_date, 89)
    
    df_subscriptions = spark.table("silver.subscriptions_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1)
    
    # 8. service_address_change_90d_flag
    # 9. broadband_upgrade_90d_flag
    df_service_events = (
        df_subscriptions
        .withColumn("service_address_change_90d_flag",
                   F.when((F.col("address_change_date").isNotNull()) & 
                          (F.col("address_change_date") >= start_date_90d) &
                          (F.col("address_change_date") <= snapshot_date), 1)
                    .otherwise(0))
        .withColumn("broadband_upgrade_90d_flag",
                   F.when((F.col("upgrade_event_flag") == 1) &
                          (F.col("upgrade_date") >= start_date_90d) &
                          (F.col("upgrade_date") <= snapshot_date), 1)
                    .otherwise(0))
        .groupBy("customer_id")
        .agg(
            F.max("service_address_change_90d_flag").alias("service_address_change_90d_flag"),
            F.max("broadband_upgrade_90d_flag").alias("broadband_upgrade_90d_flag")
        )
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_service_events

def calculate_presence_mobility_features(spark, snapshot_month, data_month):
    """NHÓM 5: HIỆN DIỆN/VẮNG NHÀ"""
    print("🔍 Calculating presence & mobility features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_30d = F.date_sub(snapshot_date, 29)
    
    df_customers = get_customer_base(spark, data_month)
    df_usage = spark.table("silver.usage_silver")
    df_cdr = spark.table("silver.cdr_silver")
    df_msisdn_mapping = get_msisdn_to_customer_mapping(spark, data_month)
    
    # Date range cho partition filter
    start_date_py = datetime.strptime(data_month + '01', '%Y%m%d').date()
    date_range = [(start_date_py - timedelta(days=x)).strftime('%Y%m%d') for x in range(30, 0, -1)]
    
    # 10. home_absence_days_from_usage_30d
    # Tính lưu lượng theo ngày và xác định khung giờ hoạt động
    df_hourly_usage = (
        df_usage
        .filter(F.col("date").isin(date_range))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("time_block",
                   F.when(F.col("hour").between(6, 9), "morning")
                    .when(F.col("hour").between(18, 24), "evening")
                    .otherwise("other"))
    )
    
    # FIX: Sửa lỗi cộng cột - dùng F.col() cho từng cột
    df_activity_blocks = (
        df_hourly_usage
        .filter(F.col("time_block").isin(["morning", "evening"]))
        .groupBy("customer_id", "date_stamp", "time_block")
        .agg(F.sum(F.col("uplink_bytes") + F.col("downlink_bytes")).alias("block_bytes"))
        .withColumn("has_activity", F.when(F.col("block_bytes") > 50 * 1024 * 1024, 1).otherwise(0))  # 50MB threshold
        .groupBy("customer_id", "date_stamp")
        .agg(F.sum("has_activity").alias("active_blocks"))
    )
    
    # FIX: Sửa lỗi tương tự trong daily usage
    df_daily_usage = (
        df_usage
        .filter(F.col("date").isin(date_range))
        .groupBy("customer_id", "date_stamp")
        .agg(F.sum(F.col("uplink_bytes") + F.col("downlink_bytes")).alias("daily_bytes"))
        .withColumn("daily_mb", F.col("daily_bytes") / (1024 * 1024))
        .join(df_activity_blocks, ["customer_id", "date_stamp"], "left")
        .fillna(0)
        .withColumn("is_absent", 
                   F.when((F.col("daily_mb") < 200) & (F.col("active_blocks") < 2), 1)
                    .otherwise(0))
    )
    
    df_absence_days = (
        df_daily_usage
        .groupBy("customer_id")
        .agg(F.sum("is_absent").alias("home_absence_days_from_usage_30d"))
    )
    
    # 11. away_days_from_cdr_30d & 12. multi_area_mobility_days_30d
    df_cdr_30d = (
        df_cdr
        .filter(F.col("date").isin(date_range))
        .withColumn("hour", F.hour("timestamp"))
    )
    
    df_night_cdr = (
        df_cdr_30d
        .filter(F.col("hour").between(21, 23) | F.col("hour").between(0, 6))
    )
    
    df_night_cdr_mapped = (
        df_night_cdr
        .join(df_msisdn_mapping, "msisdn", "inner")
        .select("customer_id", "cell_id", "timestamp")
    )
    
    # Xác định home cell cluster (14-30 ngày gần nhất)
    df_home_period = (
        df_night_cdr_mapped
        .withColumn("date_stamp", F.to_date("timestamp"))
        .filter(F.col("date_stamp") >= F.date_sub(snapshot_date, 29))  # 30 ngày
        .groupBy("customer_id", "cell_id")
        .agg(F.count("*").alias("appearance_count"))
    )
    
    window_spec = Window.partitionBy("customer_id").orderBy(F.desc("appearance_count"))
    df_home_cluster = (
        df_home_period
        .withColumn("rank", F.row_number().over(window_spec))
        .filter(F.col("rank") == 1)
        .select("customer_id", F.col("cell_id").alias("home_cell_id"))
    )
    
    # Tính away days và multi-area days
    df_daily_cdr_analysis = (
        df_cdr_30d
        .join(df_msisdn_mapping, "msisdn", "inner")
        .select("customer_id", "cell_id", "timestamp")
        .withColumn("date_stamp", F.to_date("timestamp"))
        .join(df_home_cluster, "customer_id", "left")
        .withColumn("is_away", F.when(F.col("cell_id") != F.col("home_cell_id"), 1).otherwise(0))
        .groupBy("customer_id", "date_stamp")
        .agg(
            F.max("is_away").alias("daily_away"),
            F.count_distinct("cell_id").alias("unique_cells")
        )
        .withColumn("is_multi_area", F.when(F.col("unique_cells") >= 2, 1).otherwise(0))
    )
    
    df_mobility_days = (
        df_daily_cdr_analysis
        .groupBy("customer_id")
        .agg(
            F.sum("daily_away").alias("away_days_from_cdr_30d"),
            F.sum("is_multi_area").alias("multi_area_mobility_days_30d")
        )
    )
    
    df_result = (
        df_customers
        .join(df_absence_days, "customer_id", "left")
        .join(df_mobility_days, "customer_id", "left")
        .fillna(0)
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_result

def calculate_technical_support_features(spark, snapshot_month, data_month):
    """NHÓM 6: HỖ TRỢ KỸ THUẬT"""
    print("🔍 Calculating technical support features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_90d = F.date_sub(snapshot_date, 89)
    start_date_180d = F.date_sub(snapshot_date, 179)
    
    df_customers = get_customer_base(spark, data_month)
    df_tickets = spark.table("silver.tickets_silver")
    
    # 13. wifi_coverage_ticket_cnt_90d - với deduplication
    wifi_topics = ["wifi_coverage", "weak_signal", "router_position"]
    
    df_wifi_tickets_dedup = (
        df_tickets
        .filter((F.col("create_time") >= start_date_90d) & (F.col("create_time") <= snapshot_date))
        .filter(F.lower(F.col("topic_group")).isin([t.lower() for t in wifi_topics]))
        .withColumn("date_trunc", F.date_trunc("day", "create_time"))
        .groupBy("customer_id", "date_trunc", "topic_group")
        .agg(F.first("ticket_id").alias("ticket_id"))  # Deduplicate per day per topic
    )
    
    df_wifi_tickets = (
        df_wifi_tickets_dedup
        .groupBy("customer_id")
        .agg(F.count("ticket_id").alias("wifi_coverage_ticket_cnt_90d"))
    )
    
    # 14. security_issue_ticket_cnt_180d
    security_topics = ["security_alarm", "motion_alert", "bell_issue", "camera_alarm"]
    
    df_security_tickets = (
        df_tickets
        .filter((F.col("create_time") >= start_date_180d) & (F.col("create_time") <= snapshot_date))
        .filter(F.lower(F.col("topic_group")).isin([t.lower() for t in security_topics]))
        .groupBy("customer_id")
        .agg(F.count("ticket_id").alias("security_issue_ticket_cnt_180d"))
    )
    
    df_result = (
        df_customers
        .join(df_wifi_tickets, "customer_id", "left")
        .join(df_security_tickets, "customer_id", "left")
        .fillna(0)
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_result

def calculate_business_features(spark, snapshot_month, data_month):
    """NHÓM 7: KINH DOANH"""
    print("🔍 Calculating business features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_90d = F.date_sub(snapshot_date, 89)
    start_date_180d = F.date_sub(snapshot_date, 179)
    
    df_customers = get_customer_base(spark, data_month)
    df_tickets = spark.table("silver.tickets_silver")
    df_subscriptions = spark.table("silver.subscriptions_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1)
    df_service_profile = spark.table("silver.service_profile_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1)
    df_usage = spark.table("silver.usage_silver")
    
    # 15. retail_peak_usage_pattern_90d_flag
    # Tính usage pattern cho retail (16-22h và cuối tuần)
    start_date_py = datetime.strptime(data_month + '01', '%Y%m%d').date()
    date_range_90d = [(start_date_py - timedelta(days=x)).strftime('%Y%m%d') for x in range(90, 0, -1)]
    
    df_usage_90d = (
        df_usage
        .filter(F.col("date").isin(date_range_90d))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))  # Sun=1, Sat=7
        .withColumn("is_peak_hour", F.when(F.col("hour").between(16, 22), 1).otherwise(0))
        .withColumn("total_bytes", F.col("uplink_bytes") + F.col("downlink_bytes"))
    )
    
    # Tính baseline (ngày thường không phải giờ cao điểm)
    df_baseline = (
        df_usage_90d
        .filter((F.col("is_weekend") == 0) & (F.col("is_peak_hour") == 0))
        .groupBy("customer_id")
        .agg(F.avg("total_bytes").alias("avg_baseline_bytes"))
    )
    
    # Tính peak usage (giờ cao điểm và cuối tuần)
    df_peak_usage = (
        df_usage_90d
        .filter((F.col("is_weekend") == 1) | (F.col("is_peak_hour") == 1))
        .groupBy("customer_id", "date_stamp")
        .agg(F.sum("total_bytes").alias("daily_peak_bytes"))
        .join(df_baseline, "customer_id", "left")
        .withColumn("peak_ratio", F.col("daily_peak_bytes") / F.col("avg_baseline_bytes"))
        .withColumn("is_peak_day", F.when(F.col("peak_ratio") > 1.5, 1).otherwise(0))  # >50% increase
        .groupBy("customer_id")
        .agg(F.avg("is_peak_day").alias("peak_day_ratio"))
        .withColumn("retail_peak_usage_pattern_90d_flag",
                   F.when(F.col("peak_day_ratio") > 0.5, 1).otherwise(0))  # >50% days show peak pattern
    )
    
    # 16. pos_related_ticket_cnt_180d
    pos_topics = ["pos", "people_counting", "store_camera"]
    
    df_pos_tickets = (
        df_tickets
        .filter((F.col("create_time") >= start_date_180d) & (F.col("create_time") <= snapshot_date))
        .filter(F.lower(F.col("topic_group")).isin([t.lower() for t in pos_topics]))
        .groupBy("customer_id")
        .agg(F.count("ticket_id").alias("pos_related_ticket_cnt_180d"))
    )
    
    # 17. multi_site_service_count
    df_multi_site = (
        df_subscriptions
        .groupBy("customer_id")
        .agg(F.count_distinct("service_id").alias("multi_site_service_count"))
    )
    
    # 18. avg_daily_wifi_clients_30d
    # Giả sử wifi_clients_count_daily là string chứa số lượng, cần parse
    date_range_30d = [(start_date_py - timedelta(days=x)).strftime('%Y%m%d') for x in range(30, 0, -1)]
    
    df_wifi_clients = (
        df_service_profile
        .filter(F.col("wifi_clients_count_daily").isNotNull())
        .withColumn("wifi_clients", F.col("wifi_clients_count_daily").cast("double"))
        .filter(F.col("wifi_clients").isNotNull())
        .groupBy("customer_id")
        .agg(F.avg("wifi_clients").alias("avg_daily_wifi_clients_30d"))
    )
    
    df_result = (
        df_customers
        .join(df_peak_usage.select("customer_id", "retail_peak_usage_pattern_90d_flag"), "customer_id", "left")
        .join(df_pos_tickets, "customer_id", "left")
        .join(df_multi_site, "customer_id", "left")
        .join(df_wifi_clients, "customer_id", "left")
        .fillna(0)
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_result

def calculate_time_sensitive_features(spark, snapshot_month, data_month):
    """NHÓM 8: THỜI GIAN NHẠY CẢM"""
    print("🔍 Calculating time sensitive features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_30d = F.date_sub(snapshot_date, 29)
    
    df_customers = spark.table("silver.crm_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1) \
        .select("customer_id", "ward").distinct()
    
    df_subscriptions = spark.table("silver.subscriptions_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1)
    
    df_tickets = spark.table("silver.tickets_silver")
    
    # 19. post_movein_period_flag
    df_recent_movers = (
        df_subscriptions
        .filter(F.col("address_change_date").isNotNull())
        .withColumn("days_since_move", F.datediff(snapshot_date, "address_change_date"))
        .withColumn("post_movein_period_flag", 
                   F.when((F.col("days_since_move") >= 14) & (F.col("days_since_move") <= 56), 1)
                    .otherwise(0))
        .select("customer_id", "post_movein_period_flag")
    )
    
    # 20. neighbor_security_incident_flag
    df_recent_security_tickets = (
        df_tickets
        .filter((F.col("create_time") >= start_date_30d) & (F.col("create_time") <= snapshot_date))
        .filter(F.lower(F.col("topic_group")).isin(["security_issue", "theft", "intrusion"]))
        .join(df_customers.select("customer_id", "ward"), "customer_id", "inner")
        .groupBy("ward")
        .agg(F.count("ticket_id").alias("recent_security_count"))
        .withColumn("has_recent_incident", F.when(F.col("recent_security_count") >= 1, 1).otherwise(0))
        .select("ward", "has_recent_incident")
    )
    
    df_result = (
        df_customers
        .join(df_recent_movers, "customer_id", "left")
        .join(df_recent_security_tickets, "ward", "left")
        .withColumn("neighbor_security_incident_flag", 
                   F.coalesce(F.col("has_recent_incident"), F.lit(0)))
        .fillna(0, ["post_movein_period_flag"])
        .select("customer_id", "post_movein_period_flag", "neighbor_security_incident_flag")
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_result

def calculate_demographic_features(spark, snapshot_month, data_month):
    """NHÓM 9: NHÂN KHẨU HỌC"""
    print("🔍 Calculating demographic features...")
    
    snapshot_date = F.last_day(F.to_date(F.lit(data_month + '01'), 'yyyyMMdd'))
    start_date_90d = F.date_sub(snapshot_date, 89)
    
    df_crm = spark.table("silver.crm_silver") \
        .filter(F.col("month") == data_month) \
        .filter(F.col("is_current") == 1)
    
    # 21. household_has_children_flag
    # 22. household_has_elderly_flag  
    # 23. household_composition_change_90d_flag
    df_demographics = (
        df_crm
        .select("customer_id", "household_member_age", "household_change_date")
        .distinct()
        .withColumn("has_children", 
                   F.when(F.col("household_member_age").isNotNull(),
                         F.exists(F.col("household_member_age"), lambda x: x < 12))  # <12 tuổi theo spec
                    .otherwise(False))
        .withColumn("has_elderly",
                   F.when(F.col("household_member_age").isNotNull(),
                         F.exists(F.col("household_member_age"), lambda x: x >= 65))
                    .otherwise(False))
        .withColumn("household_has_children_flag", 
                   F.when(F.col("has_children"), 1).otherwise(0))
        .withColumn("household_has_elderly_flag", 
                   F.when(F.col("has_elderly"), 1).otherwise(0))
        .withColumn("household_composition_change_90d_flag",
                   F.when((F.col("household_change_date").isNotNull()) &
                          (F.col("household_change_date") >= start_date_90d) &
                          (F.col("household_change_date") <= snapshot_date), 1)
                    .otherwise(0))
        .select("customer_id",
                "household_has_children_flag",
                "household_has_elderly_flag", 
                "household_composition_change_90d_flag")
        .withColumn("snapshot_month", F.lit(snapshot_month))
    )
    
    return df_demographics

def create_complete_feature_dataframe(spark, snapshot_month):
    """Tạo DataFrame hoàn chỉnh với tất cả 9 nhóm features"""
    data_month = get_previous_month(snapshot_month)
    print(f"🎯 Creating features for snapshot {snapshot_month} using data from {data_month}")
    
    # Tính toán tất cả 9 nhóm features
    df_direct_intent = calculate_direct_intent_features(spark, snapshot_month, data_month)
    df_usage_pattern = calculate_usage_pattern_features(spark, snapshot_month, data_month)
    df_housing_context = calculate_housing_context_features(spark, snapshot_month, data_month)
    df_service_events = calculate_service_event_features(spark, snapshot_month, data_month)
    df_presence_mobility = calculate_presence_mobility_features(spark, snapshot_month, data_month)
    df_technical_support = calculate_technical_support_features(spark, snapshot_month, data_month)
    df_business = calculate_business_features(spark, snapshot_month, data_month)
    df_time_sensitive = calculate_time_sensitive_features(spark, snapshot_month, data_month)
    df_demographics = calculate_demographic_features(spark, snapshot_month, data_month)
    
    # Lấy customer base làm foundation
    df_customers = get_customer_base(spark, data_month)
    
    print(f"📊 Base customer count: {df_customers.count()}")
    
    # JOIN TẤT CẢ 9 NHÓM FEATURES
    print("🔄 Joining all 9 feature groups...")
    
    df_all_features = (
        df_customers
        .withColumn("snapshot_month", F.lit(snapshot_month))
        .withColumn("month", F.lit(snapshot_month))
        # Join từng nhóm features
        .join(df_direct_intent.select(
            "customer_id", 
            "camera_install_inquiry_30d_flag",
            "camera_security_ticket_cnt_90d", 
            "browsing_camera_topic_hits_30d"
        ), "customer_id", "left")
        .join(df_usage_pattern.select(
            "customer_id",
            "night_small_ul_streaksum_30d",
            "small_ul_all_day_streaksum_30d"
        ), "customer_id", "left")
        .join(df_housing_context.select(
            "customer_id",
            "housing_type_street_house_flag",
            "corner_house_probability_score"
        ), "customer_id", "left")
        .join(df_service_events.select(
            "customer_id",
            "service_address_change_90d_flag",
            "broadband_upgrade_90d_flag"
        ), "customer_id", "left")
        .join(df_presence_mobility.select(
            "customer_id",
            "home_absence_days_from_usage_30d",
            "away_days_from_cdr_30d", 
            "multi_area_mobility_days_30d"
        ), "customer_id", "left")
        .join(df_technical_support.select(
            "customer_id",
            "wifi_coverage_ticket_cnt_90d",
            "security_issue_ticket_cnt_180d"
        ), "customer_id", "left")
        .join(df_business.select(
            "customer_id",
            "retail_peak_usage_pattern_90d_flag",
            "pos_related_ticket_cnt_180d",
            "multi_site_service_count",
            "avg_daily_wifi_clients_30d"
        ), "customer_id", "left")
        .join(df_time_sensitive.select(
            "customer_id",
            "post_movein_period_flag", 
            "neighbor_security_incident_flag"
        ), "customer_id", "left")
        .join(df_demographics.select(
            "customer_id",
            "household_has_children_flag",
            "household_has_elderly_flag",
            "household_composition_change_90d_flag"
        ), "customer_id", "left")
        .withColumn("updated_at", F.current_timestamp())
        .withColumn("processed_date", F.current_date())
        .fillna(0)
        .distinct()
    )
    
    final_count = df_all_features.count()
    distinct_count = df_all_features.select("customer_id").distinct().count()
    
    print(f"✅ FINAL RESULT: {final_count} records for {distinct_count} customers")
    print(f"✅ ALL 9 FEATURE GROUPS COMPLETED for snapshot {snapshot_month}")
    
    return df_all_features

def main():
    """Main function để chạy silver-to-gold transformation"""
    ap = argparse.ArgumentParser(description="Silver to Gold Feature Engineering")
    ap.add_argument("--month", type=str, required=True, help="YYYYMM - snapshot month")
    args = ap.parse_args()

    print(f"🚀 Starting Silver-to-Gold transformation for snapshot month: {args.month}")
    
    spark = get_spark("silver-to-gold")
    
    try:
        # Bước 1: Tạo gold tables
        print("📁 Step 1: Creating gold tables...")
        create_feature_tables(spark)
        
        # Bước 2: Tạo DataFrame hoàn chỉnh
        print("🧮 Step 2: Creating complete feature DataFrame...")
        df_all_features = create_complete_feature_dataframe(spark, args.month)
        
        # Bước 3: Ghi vào gold layer
        print("💾 Step 3: Writing to gold layer...")
        
        (df_all_features
         .write
         .mode("overwrite")
         .option("partitionOverwriteMode", "dynamic")
         .saveAsTable("gold.customer_features_monthly"))
        
        print("✅ Silver-to-Gold transformation completed successfully!")
        
        # Bước 4: Kiểm tra final table
        print("🔎 Step 4: Verifying gold table...")
        
        gold_data = spark.sql(f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT customer_id) as distinct_customers
            FROM gold.customer_features_monthly 
            WHERE month = '{args.month}'
        """).collect()[0]
        
        print(f"✅ Gold table verified:")
        print(f"   • Total records: {gold_data['total_records']}")
        print(f"   • Distinct customers: {gold_data['distinct_customers']}")
        
    except Exception as e:
        print(f"❌ Error in Silver-to-Gold transformation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        spark.stop()

if __name__ == "__main__":
    main()