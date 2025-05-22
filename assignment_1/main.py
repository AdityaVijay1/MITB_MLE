import os
import glob
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
import utils.data_processing_feature_bronze_table
import utils.data_processing_feature_silver_table
import utils.data_processing_feature_gold_table


def main():
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("feature_store_pipeline") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # Set up config
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"

    # Generate list of dates to process
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    print("Processing dates:", dates_str_lst)

    # Create directory structure
    create_directories()

    # Process Label Store
    process_label_store(spark, dates_str_lst)

    # Process Feature Store
    process_feature_store(spark, dates_str_lst)

    print("Joining feature and label store to check results")

    # Join feature store and label store to validate completeness
    validate_joined_data(spark)

    print("Pipeline completed successfully!")


def generate_first_of_month_dates(start_date_str, end_date_str):
    """Generate a list of first-of-month dates between start and end dates"""
    # Convert the date strings to datetime objects
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


def create_directories():
    """Create all necessary directories for the data pipeline"""
    # Label store directories
    directories = [
        "datamart/bronze/lms/",
        "datamart/silver/loan_daily/",
        "datamart/gold/label_store/",
        
        # Feature store directories
        "datamart/bronze/clickstream/",
        "datamart/bronze/attributes/",
        "datamart/bronze/financials/",
        "datamart/bronze/loan/",  
        "datamart/silver/clickstream/",
        "datamart/silver/attributes/",
        "datamart/silver/financials/",
        "datamart/silver/loan/",  
        "datamart/gold/feature_store/"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def process_label_store(spark, dates_str_lst):
    """Process all label store components"""
    print("\n--- Processing Label Store ---")
    
    # Directory paths
    bronze_lms_directory = "datamart/bronze/lms/"
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    gold_label_store_directory = "datamart/gold/label_store/"
    
    # Run bronze backfill for label store
    print("\nProcessing Bronze Label Store Tables:")
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)
    
    # Run silver backfill for label store
    print("\nProcessing Silver Label Store Tables:")
    for date_str in dates_str_lst:
        utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    
    # Run gold backfill for label store
    print("\nProcessing Gold Label Store Tables:")
    for date_str in dates_str_lst:
        utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6)


def process_feature_store(spark, dates_str_lst):
    """Process all feature store components"""
    print("\n--- Processing Feature Store ---")
    
    # Bronze directory paths
    bronze_clickstream_directory = "datamart/bronze/clickstream/"
    bronze_attributes_directory = "datamart/bronze/attributes/"
    bronze_financials_directory = "datamart/bronze/financials/"
    bronze_loan_directory = "datamart/bronze/loan/"  # Added loan directory
    
    # Silver directory paths
    silver_clickstream_directory = "datamart/silver/clickstream/"
    silver_attributes_directory = "datamart/silver/attributes/"
    silver_financials_directory = "datamart/silver/financials/"
    silver_loan_directory = "datamart/silver/loan/"  # Added loan directory
    
    # Gold directory path
    gold_feature_store_directory = "datamart/gold/feature_store/"
    
    # Run bronze backfill for feature store
    print("\nProcessing Bronze Feature Store Tables:")
    for date_str in dates_str_lst:
        utils.data_processing_feature_bronze_table.process_bronze_clickstream(date_str, bronze_clickstream_directory, spark)
        utils.data_processing_feature_bronze_table.process_bronze_attributes(date_str, bronze_attributes_directory, spark)
        utils.data_processing_feature_bronze_table.process_bronze_financials(date_str, bronze_financials_directory, spark)
        utils.data_processing_feature_bronze_table.process_bronze_loan(date_str, bronze_loan_directory, spark)
    
    # Run silver backfill for feature store
    print("\nProcessing Silver Feature Store Tables:")
    for date_str in dates_str_lst:
        utils.data_processing_feature_silver_table.process_silver_clickstream(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)
        utils.data_processing_feature_silver_table.process_silver_attributes(date_str, bronze_attributes_directory, silver_attributes_directory, spark)
        utils.data_processing_feature_silver_table.process_silver_financials(date_str, bronze_financials_directory, silver_financials_directory, spark)
        utils.data_processing_feature_silver_table.process_silver_loan(date_str, bronze_loan_directory, silver_loan_directory, spark)  
        
    # Run gold backfill for feature store
    print("\nProcessing Gold Feature Store Tables:")
    for date_str in dates_str_lst:
        utils.data_processing_feature_gold_table.process_gold_feature_store(
            date_str,
            silver_clickstream_directory,
            silver_attributes_directory,
            silver_financials_directory,
            silver_loan_directory,  
            gold_feature_store_directory,
            spark
        )


def validate_joined_data(spark):
    """
    Join feature store and label store to validate data completeness and
    ensure there's no data leakage.
    """
    from pyspark.sql.functions import col  
    
    print("\n--- Validating Feature Store and Label Store Integration ---")
    
    # Get all gold feature store files
    gold_feature_store_directory = "datamart/gold/feature_store/"
    feature_files = glob.glob(os.path.join(gold_feature_store_directory, "*.parquet"))
    if not feature_files:
        print("No feature store files found!")
        return
        
    # Get all gold label store files
    gold_label_store_directory = "datamart/gold/label_store/"
    label_files = glob.glob(os.path.join(gold_label_store_directory, "*.parquet"))
    if not label_files:
        print("No label store files found!")
        return
    
    # Read feature store data
    feature_store_df = spark.read.parquet(*feature_files)
    feature_count = feature_store_df.count()
    feature_columns = len(feature_store_df.columns)
    print(f"Feature store: {feature_count} rows, {feature_columns} columns")
    
    # Read label store data
    label_store_df = spark.read.parquet(*label_files)
    label_count = label_store_df.count()
    print(f"Label store: {label_count} rows")
    
    
    feature_cols = set(feature_store_df.columns)
    label_cols = set(label_store_df.columns)
    common_cols = feature_cols.intersection(label_cols)
    print(f"Common columns between feature and label stores: {common_cols}")
    
    # Rename conflicting columns in label store (except join keys)
    join_keys = ["Customer_ID", "snapshot_date"]
    label_store_renamed = label_store_df
    for col_name in common_cols:
        if col_name not in join_keys:
            label_store_renamed = label_store_renamed.withColumnRenamed(col_name, f"label_{col_name}")
    
    # Join feature and label data on Customer_ID and snapshot_date
    joined_df = feature_store_df.join(
        label_store_renamed,
        on=join_keys,
        how="inner"
    )
    
    joined_count = joined_df.count()
    print(f"Joined data: {joined_count} rows")
    
    
    mob6_records = feature_store_df.filter(col("mob") == 6).count()
    print(f"Records with mob=6 in feature store: {mob6_records}")
    
    
    loan_id_records = feature_store_df.filter(col("loan_id").isNotNull()).count()
    print(f"Records with loan_id (non-null) in feature store: {loan_id_records}")
    
    # Personal check for potential data leakage
    print("\nChecking for potential data leakage...")
    leakage_terms = ["label", "default"]  
    potential_leakage_cols = [col_name for col_name in feature_store_df.columns 
                             if any(term in col_name.lower() for term in leakage_terms)]
    
    if potential_leakage_cols:
        print("WARNING: Potential data leakage detected in these columns:")
        for col_name in potential_leakage_cols:
            print(f" - {col_name}")
    else:
        print("No obvious data leakage detected in column names.")
    
    # Display sample of joined data
    print("\nSample of joined data (feature store + label store):")
    # Select columns that definitely exist and are unambiguous
    base_columns = ["Customer_ID", "snapshot_date"]
    
    # Add feature store specific columns
    feature_specific_cols = ["loan_id", "mob", "Annual_Income", "Credit_Mix_Score", "Combined_Risk_Score"]
    for col_name in feature_specific_cols:
        if col_name in feature_store_df.columns:
            base_columns.append(col_name)
    
    # Add label store columns (with renamed prefix)
    if "label_label" in joined_df.columns:
        base_columns.append("label_label")
    elif "label" in joined_df.columns:
        base_columns.append("label")
        
    if "label_label_def" in joined_df.columns:
        base_columns.append("label_label_def")
    elif "label_def" in joined_df.columns:
        base_columns.append("label_def")
    
    # Show sample with available columns
    available_columns = [col_name for col_name in base_columns if col_name in joined_df.columns]
    print(f"Showing columns: {available_columns}")
    
    if available_columns:
        joined_df.select(*available_columns).show(5)
    else:
        print("No suitable columns found for display")
        print(f"All joined dataframe columns: {joined_df.columns}")
    
    print("\nJoin validation complete!")


if __name__ == "__main__":
    main()