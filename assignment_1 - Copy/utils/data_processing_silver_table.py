from pyspark.sql.functions import col, to_date, datediff, add_months, ceil, lit, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_clickstream_silver(date_str, spark):
    """Process clickstream data into silver table"""
    # Read bronze clickstream data
    input_path = f"datamart/bronze/clickstream/clickstream_{date_str.replace('-', '_')}.parquet"
    df = spark.read.parquet(input_path)
    print(f'Loaded clickstream from: {input_path}')
    
    # Clean and transform data
    df = df.withColumn("session_duration", col("session_duration").cast(IntegerType()))
    df = df.withColumn("page_views", col("page_views").cast(IntegerType()))
    df = df.withColumn("time_on_site", col("time_on_site").cast(IntegerType()))
    
    # Save to silver layer
    output_path = f"datamart/silver/clickstream/clickstream_{date_str.replace('-', '_')}.parquet"
    df.write.mode("overwrite").parquet(output_path)
    print(f'Saved clickstream to: {output_path}')
    return df

def process_attributes_silver(date_str, spark):
    """Process attributes data into silver table"""
    # Read bronze attributes data
    input_path = f"datamart/bronze/attributes/attributes_{date_str.replace('-', '_')}.parquet"
    df = spark.read.parquet(input_path)
    print(f'Loaded attributes from: {input_path}')
    
    # Clean and transform data
    df = df.withColumn("age", col("age").cast(IntegerType()))
    df = df.withColumn("income", col("income").cast(FloatType()))
    df = df.withColumn("credit_score", col("credit_score").cast(IntegerType()))
    
    # Save to silver layer
    output_path = f"datamart/silver/attributes/attributes_{date_str.replace('-', '_')}.parquet"
    df.write.mode("overwrite").parquet(output_path)
    print(f'Saved attributes to: {output_path}')
    return df

def process_financials_silver(date_str, spark):
    """Process financials data into silver table"""
    # Read bronze financials data
    input_path = f"datamart/bronze/financials/financials_{date_str.replace('-', '_')}.parquet"
    df = spark.read.parquet(input_path)
    print(f'Loaded financials from: {input_path}')
    
    # Clean and transform data
    df = df.withColumn("total_assets", col("total_assets").cast(FloatType()))
    df = df.withColumn("total_liabilities", col("total_liabilities").cast(FloatType()))
    df = df.withColumn("monthly_income", col("monthly_income").cast(FloatType()))
    df = df.withColumn("monthly_expenses", col("monthly_expenses").cast(FloatType()))
    
    # Calculate derived metrics
    df = df.withColumn("net_worth", col("total_assets") - col("total_liabilities"))
    df = df.withColumn("savings_rate", (col("monthly_income") - col("monthly_expenses")) / col("monthly_income"))
    
    # Save to silver layer
    output_path = f"datamart/silver/financials/financials_{date_str.replace('-', '_')}.parquet"
    df.write.mode("overwrite").parquet(output_path)
    print(f'Saved financials to: {output_path}')
    return df

def process_loan_daily_silver(date_str, spark):
    """Process loan daily data into silver table"""
    # Read bronze loan daily data
    input_path = f"datamart/bronze/loan_daily/loan_daily_{date_str.replace('-', '_')}.parquet"
    df = spark.read.parquet(input_path)
    print(f'Loaded loan daily from: {input_path}')
    
    # Clean and transform data
    df = df.withColumn("loan_id", col("loan_id").cast(StringType()))
    df = df.withColumn("customer_id", col("customer_id").cast(StringType()))
    df = df.withColumn("loan_start_date", col("loan_start_date").cast(DateType()))
    df = df.withColumn("tenure", col("tenure").cast(IntegerType()))
    df = df.withColumn("installment_num", col("installment_num").cast(IntegerType()))
    df = df.withColumn("loan_amt", col("loan_amt").cast(FloatType()))
    df = df.withColumn("due_amt", col("due_amt").cast(FloatType()))
    df = df.withColumn("paid_amt", col("paid_amt").cast(FloatType()))
    df = df.withColumn("overdue_amt", col("overdue_amt").cast(FloatType()))
    df = df.withColumn("balance", col("balance").cast(FloatType()))
    
    # Calculate derived metrics
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn("installments_missed", ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType()))
    df = df.withColumn("first_missed_date", 
                      when(col("installments_missed") > 0, 
                           add_months(col("snapshot_date"), -1 * col("installments_missed")))
                      .cast(DateType()))
    df = df.withColumn("dpd", 
                      when(col("overdue_amt") > 0.0, 
                           datediff(col("snapshot_date"), col("first_missed_date")))
                      .otherwise(0)
                      .cast(IntegerType()))
    
    # Save to silver layer
    output_path = f"datamart/silver/loan_daily/loan_daily_{date_str.replace('-', '_')}.parquet"
    df.write.mode("overwrite").parquet(output_path)
    print(f'Saved loan daily to: {output_path}')
    return df

def process_silver_tables(date_str, spark):
    """Process all silver tables for a given date"""
    clickstream_df = process_clickstream_silver(date_str, spark)
    attributes_df = process_attributes_silver(date_str, spark)
    financials_df = process_financials_silver(date_str, spark)
    loan_daily_df = process_loan_daily_silver(date_str, spark)
    
    return {
        "clickstream": clickstream_df,
        "attributes": attributes_df,
        "financials": financials_df,
        "loan_daily": loan_daily_df
    }