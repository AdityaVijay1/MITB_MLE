import os
import glob
import pandas as pd
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_clickstream(snapshot_date_str, bronze_clickstream_directory, spark):
    """
    Process clickstream data into bronze layer
    This is an iterative/incremental table - new data is added
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    csv_file_path = "data/feature_clickstream.csv"

    # load data
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # Convert snapshot_date column to proper date format
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "M/d/yyyy"))
    
    # Filter for current snapshot date
    df = df.filter(col('snapshot_date') == snapshot_date)
    
    print(f"Clickstream {snapshot_date_str} row count:", df.count())

    # save bronze table to datamart
    partition_name = f"bronze_clickstream_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_clickstream_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def process_bronze_attributes(snapshot_date_str, bronze_attributes_directory, spark):
    """
    Process attributes data into bronze layer
    This is an overwrite table - new data overwrites old records
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/features_attributes.csv"

    # load data
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # Convert snapshot_date column to proper date format
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "M/d/yyyy"))
    
    # Filter for current snapshot date
    df = df.filter(col('snapshot_date') <= snapshot_date)
    
    # Select only the latest record for each customer (overwrite approach)
    window_spec = pyspark.sql.Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
    df = df.withColumn("row_num", F.row_number().over(window_spec))
    df = df.filter(col("row_num") == 1).drop("row_num")
    
    
    df = df.withColumn("processing_date", lit(snapshot_date))
    
    print(f"Attributes {snapshot_date_str} row count:", df.count())

    # save bronze table to datamart
    partition_name = f"bronze_attributes_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_attributes_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def process_bronze_financials(snapshot_date_str, bronze_financials_directory, spark):
    """
    Process financials data into bronze layer
    This is an overwrite table - new data overwrites old records
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end 
    csv_file_path = "data/features_financials.csv"

    # load data
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # Convert snapshot_date column to proper date format
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "M/d/yyyy"))
    
    # Filter for current snapshot date
    df = df.filter(col('snapshot_date') <= snapshot_date)
    
    # Select only the latest record for each customer (overwrite approach)
    window_spec = pyspark.sql.Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
    df = df.withColumn("row_num", F.row_number().over(window_spec))
    df = df.filter(col("row_num") == 1).drop("row_num")
    
    # Add current snapshot date for consistency
    df = df.withColumn("processing_date", lit(snapshot_date))
    
    print(f"Financials {snapshot_date_str} row count:", df.count())

    # save bronze table to datamart 
    partition_name = f"bronze_financials_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_financials_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def process_bronze_loan(snapshot_date_str, bronze_loan_directory, spark):
    """
    Process loan data into bronze layer
    This is an iterative/incremental table - new data is added
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end  
    csv_file_path = "data/lms_loan_daily.csv"

    # load data
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # Convert snapshot_date column to proper date format
    df = df.withColumn("snapshot_date", to_date(col("snapshot_date"), "M/d/yyyy"))
    
    # Filter for current snapshot date
    df = df.filter(col('snapshot_date') == snapshot_date)
    
    print(f"Loan {snapshot_date_str} row count:", df.count())

    # save bronze table to datamart 
    partition_name = f"bronze_loan_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_loan_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df