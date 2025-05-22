import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import pyspark
from pyspark.sql.functions import col

# Import utils modules
from utils.data_processing_bronze_table import process_bronze_tables
from utils.data_processing_silver_table import process_silver_tables
from utils.data_processing_gold_table import process_gold_tables

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("ETL Pipeline") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# Set up date range
start_date_str = "2023-01-01"
end_date_str = "2025-01-01"

def generate_first_of_month_dates(start_date_str, end_date_str):
    """Generate list of first day of month dates between start and end date"""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

# Generate dates to process (first of each month)
dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("\nProcessing dates:", dates_str_lst)

# Create directories
directories = [
    "datamart/bronze/clickstream",
    "datamart/bronze/attributes",
    "datamart/bronze/financials",
    "datamart/bronze/loan_daily",
    "datamart/silver/clickstream",
    "datamart/silver/attributes",
    "datamart/silver/financials",
    "datamart/silver/loan_daily",
    "datamart/gold/feature_store",
    "datamart/gold/label_store"
]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Process bronze tables
print("\n=== Processing Bronze Tables ===")
for date_str in dates_str_lst:
    print(f"Processing bronze tables for {date_str}")
    process_bronze_tables(date_str, spark)

# Process silver tables
print("\n=== Processing Silver Tables ===")
for date_str in dates_str_lst:
    print(f"Processing silver tables for {date_str}")
    process_silver_tables(date_str, spark)

# Process gold tables
print("\n=== Processing Gold Tables ===")
for date_str in dates_str_lst:
    print(f"Processing gold tables for {date_str}")
    process_gold_tables(date_str, spark)

# Verify final output
print("\n=== Verifying Final Output ===")
gold_files = glob.glob("datamart/gold/feature_store/*.parquet")
if gold_files:
    df = spark.read.parquet(*gold_files)
    print(f"Total rows in gold layer: {df.count()}")
    df.show(5)
else:
    print("No files found in gold layer")

spark.stop()