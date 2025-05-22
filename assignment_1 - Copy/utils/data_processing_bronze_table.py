from pyspark.sql.functions import col, to_date, datediff, add_months, ceil, lit, when, year, month, last_day, date_format, sum, avg, max, min
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from datetime import datetime, timedelta

def aggregate_to_monthly(df, date_str, spark):
    """Aggregate daily data to monthly data"""
    # Convert date string to datetime
    current_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Calculate the start and end of the previous month
    if current_date.month == 1:
        start_date = datetime(current_date.year - 1, 12, 1)
    else:
        start_date = datetime(current_date.year, current_date.month - 1, 1)
    
    end_date = datetime(current_date.year, current_date.month, 1) - timedelta(days=1)
    
    # Filter data for the previous month
    df = df.filter((col("snapshot_date") >= start_date) & (col("snapshot_date") <= end_date))
    
    return df

def process_clickstream_bronze(date_str, spark):
    """Process clickstream data into bronze table"""
    # Read raw clickstream data
    input_path = "data/feature_clickstream.csv"  # Read the full dataset
    df = spark.read.option("header", "true").csv(input_path)
    print(f'Loaded clickstream from: {input_path}')
    
    # Convert date columns
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))
    
    # Aggregate to monthly data
    monthly_df = aggregate_to_monthly(df, date_str, spark)
    
    # Aggregate metrics
    monthly_df = monthly_df.groupBy("customer_id").agg(
        avg("session_duration").alias("session_duration"),
        sum("page_views").alias("page_views"),
        avg("time_on_site").alias("time_on_site")
    )
    
    # Add snapshot date (first of the month)
    monthly_df = monthly_df.withColumn("snapshot_date", to_date(lit(date_str)))
    
    # Clean and transform data
    monthly_df = monthly_df.withColumn("customer_id", col("customer_id").cast(StringType()))
    monthly_df = monthly_df.withColumn("session_duration", col("session_duration").cast(IntegerType()))
    monthly_df = monthly_df.withColumn("page_views", col("page_views").cast(IntegerType()))
    monthly_df = monthly_df.withColumn("time_on_site", col("time_on_site").cast(IntegerType()))
    
    # Save to bronze layer
    output_path = f"datamart/bronze/clickstream/clickstream_{date_str.replace('-', '_')}.parquet"
    monthly_df.write.mode("overwrite").parquet(output_path)
    print(f'Saved clickstream to: {output_path}')
    return monthly_df

def process_attributes_bronze(date_str, spark):
    """Process attributes data into bronze table"""
    # Read raw attributes data
    input_path = "data/features_attributes.csv"  # Read the full dataset
    df = spark.read.option("header", "true").csv(input_path)
    print(f'Loaded attributes from: {input_path}')
    
    # Convert date columns
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))
    
    # Get the latest attributes for each customer
    monthly_df = aggregate_to_monthly(df, date_str, spark)
    monthly_df = monthly_df.orderBy("snapshot_date", ascending=False) \
        .groupBy("customer_id") \
        .agg(
            first("age").alias("age"),
            first("income").alias("income"),
            first("credit_score").alias("credit_score")
        )
    
    # Add snapshot date (first of the month)
    monthly_df = monthly_df.withColumn("snapshot_date", to_date(lit(date_str)))
    
    # Clean and transform data
    monthly_df = monthly_df.withColumn("customer_id", col("customer_id").cast(StringType()))
    monthly_df = monthly_df.withColumn("age", col("age").cast(IntegerType()))
    monthly_df = monthly_df.withColumn("income", col("income").cast(FloatType()))
    monthly_df = monthly_df.withColumn("credit_score", col("credit_score").cast(IntegerType()))
    
    # Save to bronze layer
    output_path = f"datamart/bronze/attributes/attributes_{date_str.replace('-', '_')}.parquet"
    monthly_df.write.mode("overwrite").parquet(output_path)
    print(f'Saved attributes to: {output_path}')
    return monthly_df

def process_financials_bronze(date_str, spark):
    """Process financials data into bronze table"""
    # Read raw financials data
    input_path = "data/features_financials.csv"  # Read the full dataset
    df = spark.read.option("header", "true").csv(input_path)
    print(f'Loaded financials from: {input_path}')
    
    # Convert date columns
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))
    
    # Get the latest financials for each customer
    monthly_df = aggregate_to_monthly(df, date_str, spark)
    monthly_df = monthly_df.orderBy("snapshot_date", ascending=False) \
        .groupBy("customer_id") \
        .agg(
            first("total_assets").alias("total_assets"),
            first("total_liabilities").alias("total_liabilities"),
            first("monthly_income").alias("monthly_income"),
            first("monthly_expenses").alias("monthly_expenses")
        )
    
    # Add snapshot date (first of the month)
    monthly_df = monthly_df.withColumn("snapshot_date", to_date(lit(date_str)))
    
    # Clean and transform data
    monthly_df = monthly_df.withColumn("customer_id", col("customer_id").cast(StringType()))
    monthly_df = monthly_df.withColumn("total_assets", col("total_assets").cast(FloatType()))
    monthly_df = monthly_df.withColumn("total_liabilities", col("total_liabilities").cast(FloatType()))
    monthly_df = monthly_df.withColumn("monthly_income", col("monthly_income").cast(FloatType()))
    monthly_df = monthly_df.withColumn("monthly_expenses", col("monthly_expenses").cast(FloatType()))
    
    # Save to bronze layer
    output_path = f"datamart/bronze/financials/financials_{date_str.replace('-', '_')}.parquet"
    monthly_df.write.mode("overwrite").parquet(output_path)
    print(f'Saved financials to: {output_path}')
    return monthly_df

def process_loan_daily_bronze(date_str, spark):
    """Process loan daily data into bronze table"""
    # Read raw loan daily data
    input_path = "data/lms_loan_daily.csv"  # Read the full dataset
    df = spark.read.option("header", "true").csv(input_path)
    print(f'Loaded loan daily from: {input_path}')
    
    # Convert date columns
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))
    df = df.withColumn("loan_start_date", to_date(col("loan_start_date")))
    
    # Get the latest loan status for each loan
    monthly_df = aggregate_to_monthly(df, date_str, spark)
    monthly_df = monthly_df.orderBy("snapshot_date", ascending=False) \
        .groupBy("loan_id", "customer_id") \
        .agg(
            first("loan_start_date").alias("loan_start_date"),
            first("tenure").alias("tenure"),
            first("installment_num").alias("installment_num"),
            first("loan_amt").alias("loan_amt"),
            first("due_amt").alias("due_amt"),
            first("paid_amt").alias("paid_amt"),
            first("overdue_amt").alias("overdue_amt"),
            first("balance").alias("balance")
        )
    
    # Add snapshot date (first of the month)
    monthly_df = monthly_df.withColumn("snapshot_date", to_date(lit(date_str)))
    
    # Clean and transform data
    monthly_df = monthly_df.withColumn("loan_id", col("loan_id").cast(StringType()))
    monthly_df = monthly_df.withColumn("customer_id", col("customer_id").cast(StringType()))
    monthly_df = monthly_df.withColumn("loan_start_date", col("loan_start_date").cast(DateType()))
    monthly_df = monthly_df.withColumn("tenure", col("tenure").cast(IntegerType()))
    monthly_df = monthly_df.withColumn("installment_num", col("installment_num").cast(IntegerType()))
    monthly_df = monthly_df.withColumn("loan_amt", col("loan_amt").cast(FloatType()))
    monthly_df = monthly_df.withColumn("due_amt", col("due_amt").cast(FloatType()))
    monthly_df = monthly_df.withColumn("paid_amt", col("paid_amt").cast(FloatType()))
    monthly_df = monthly_df.withColumn("overdue_amt", col("overdue_amt").cast(FloatType()))
    monthly_df = monthly_df.withColumn("balance", col("balance").cast(FloatType()))
    
    # Save to bronze layer
    output_path = f"datamart/bronze/loan_daily/loan_daily_{date_str.replace('-', '_')}.parquet"
    monthly_df.write.mode("overwrite").parquet(output_path)
    print(f'Saved loan daily to: {output_path}')
    return monthly_df

def process_bronze_tables(date_str, spark):
    """Process all bronze tables for a given date"""
    clickstream_df = process_clickstream_bronze(date_str, spark)
    attributes_df = process_attributes_bronze(date_str, spark)
    financials_df = process_financials_bronze(date_str, spark)
    loan_daily_df = process_loan_daily_bronze(date_str, spark)
    
    return {
        "clickstream": clickstream_df,
        "attributes": attributes_df,
        "financials": financials_df,
        "loan_daily": loan_daily_df
    }