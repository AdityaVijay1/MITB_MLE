from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_feature_store(date_str, spark):
    """Process data into feature store"""
    # Read silver tables
    clickstream_path = f"datamart/silver/clickstream/clickstream_{date_str.replace('-', '_')}.parquet"
    attributes_path = f"datamart/silver/attributes/attributes_{date_str.replace('-', '_')}.parquet"
    financials_path = f"datamart/silver/financials/financials_{date_str.replace('-', '_')}.parquet"
    loan_daily_path = f"datamart/silver/loan_daily/loan_daily_{date_str.replace('-', '_')}.parquet"
    
    clickstream_df = spark.read.parquet(clickstream_path)
    attributes_df = spark.read.parquet(attributes_path)
    financials_df = spark.read.parquet(financials_path)
    loan_daily_df = spark.read.parquet(loan_daily_path)
    
    print(f'Loaded silver tables for {date_str}')
    
    # Join all tables
    feature_df = loan_daily_df \
        .join(clickstream_df, ["customer_id", "snapshot_date"], "left") \
        .join(attributes_df, ["customer_id", "snapshot_date"], "left") \
        .join(financials_df, ["customer_id", "snapshot_date"], "left")
    
    # Select and rename features
    feature_df = feature_df.select(
        col("loan_id"),
        col("customer_id"),
        col("snapshot_date"),
        col("age"),
        col("income"),
        col("credit_score"),
        col("total_assets"),
        col("total_liabilities"),
        col("monthly_income"),
        col("monthly_expenses"),
        col("net_worth"),
        col("savings_rate"),
        col("session_duration"),
        col("page_views"),
        col("time_on_site"),
        col("loan_amt"),
        col("tenure"),
        col("mob"),
        col("dpd")
    )
    
    # Save to gold layer
    output_path = f"datamart/gold/feature_store/features_{date_str.replace('-', '_')}.parquet"
    feature_df.write.mode("overwrite").parquet(output_path)
    print(f'Saved feature store to: {output_path}')
    return feature_df

def process_label_store(date_str, spark):
    """Process data into label store"""
    # Read silver loan daily data
    input_path = f"datamart/silver/loan_daily/loan_daily_{date_str.replace('-', '_')}.parquet"
    df = spark.read.parquet(input_path)
    print(f'Loaded loan daily from: {input_path}')
    
    # Create labels
    df = df.filter(col("mob") == 6)  # Filter for 6 months on book
    df = df.withColumn("label", when(col("dpd") >= 30, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", lit("30dpd_6mob").cast(StringType()))
    
    # Select columns for label store
    label_df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")
    
    # Save to gold layer
    output_path = f"datamart/gold/label_store/labels_{date_str.replace('-', '_')}.parquet"
    label_df.write.mode("overwrite").parquet(output_path)
    print(f'Saved label store to: {output_path}')
    return label_df

def process_gold_tables(date_str, spark):
    """Process all gold tables for a given date"""
    feature_df = process_feature_store(date_str, spark)
    label_df = process_label_store(date_str, spark)
    
    return {
        "feature_store": feature_df,
        "label_store": label_df
    }