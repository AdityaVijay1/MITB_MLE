import os
import glob
import pandas as pd
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, isnan, isnull, split, size, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType


def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Process clickstream data into silver layer with data cleaning and transformations
    """
     
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = f"bronze_clickstream_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_clickstream_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    feature_columns = [f"fe_{i}" for i in range(1, 21)]
    
    # Cast all feature columns to Float
    for feature in feature_columns:
        df = df.withColumn(feature, col(feature).cast(FloatType()))
    
    # Cast other columns to their appropriate types
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    
    # Handle missing values with appropriate strategies
    # For numeric features, replace nulls with mean value of the column
    for feature in feature_columns:
        mean_value = df.select(F.mean(col(feature))).collect()[0][0]
        if mean_value is None:
            mean_value = 0.0
        df = df.withColumn(feature, when(isnull(col(feature)) | isnan(col(feature)), mean_value).otherwise(col(feature)))
    
    # Add additional engineered features
    df = df.withColumn("fe_sum", sum(col(f) for f in feature_columns))
    df = df.withColumn("fe_mean", df["fe_sum"] / len(feature_columns))
    
    # Create interactions between important features (example use case if fe_1 and fe_2 were the top demanding features)
    df = df.withColumn("fe_1_2_interaction", col("fe_1") * col("fe_2"))
    
    # save silver table 
    partition_name = f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet"
    filepath = os.path.join(silver_clickstream_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Process attributes data into silver layer with data cleaning and transformations
    """
     
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = f"bronze_attributes_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_attributes_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data: enforce schema / data type
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("Name", col("Name").cast(StringType()))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("SSN", col("SSN").cast(StringType()))
    df = df.withColumn("Occupation", col("Occupation").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df = df.withColumn("processing_date", col("processing_date").cast(DateType()))
    
    # Handle missing or invalid values
    # For age, replace nulls with median age
    median_age = df.approxQuantile("Age", [0.5], 0.01)[0]
    if median_age is None:
        median_age = 30
    df = df.withColumn("Age", when(isnull(col("Age")) | (col("Age") < 18) | (col("Age") > 100), median_age).otherwise(col("Age")))
    
    # Clean SSN: Filter out invalid SSNs (using regex)
    df = df.withColumn("SSN_valid", when(regexp_extract(col("SSN"), "^\\d{3}-\\d{2}-\\d{4}$", 0) != "", 1).otherwise(0))
    
    # Feature engineering
    # Create age buckets
    df = df.withColumn("Age_Bucket", 
        when(col("Age") < 25, "18-24")
        .when((col("Age") >= 25) & (col("Age") < 35), "25-34")
        .when((col("Age") >= 35) & (col("Age") < 45), "35-44")
        .when((col("Age") >= 45) & (col("Age") < 55), "45-54")
        .when((col("Age") >= 55) & (col("Age") < 65), "55-64")
        .otherwise("65+"))
    
    # One-hot encode the top occupation , for certain possible checks in the future within the model
    df = df.withColumn("is_Lawyer", when(col("Occupation") == "Lawyer", 1).otherwise(0))
    df = df.withColumn("is_Mechanic", when(col("Occupation") == "Mechanic", 1).otherwise(0))
    df = df.withColumn("is_MediaManager", when(col("Occupation") == "Media_Manager", 1).otherwise(0))
    
    # save silver table
    partition_name = f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet"
    filepath = os.path.join(silver_attributes_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Process financials data into silver layer with data cleaning and transformations
    """
     
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = f"bronze_financials_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(bronze_financials_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Clean data: enforce schema / data type and handle data quality issues
    
    # Fix Annual_Income with trailing underscore
    df = df.withColumn("Annual_Income", F.regexp_replace("Annual_Income", "_$", "").cast(DoubleType()))
    
    # Handle Type_of_Loan: Make it consistent
    df = df.withColumn("Type_of_Loan", F.when(col("Type_of_Loan").isNull() | (col("Type_of_Loan") == ""), "None").otherwise(col("Type_of_Loan")))
    
    # Handle Credit_Mix: Replace _ with "Unknown"
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix") == "_", "Unknown").otherwise(col("Credit_Mix")))
    
    # Convert Credit_History_Age to months (numeric)
    df = df.withColumn(
        "Credit_History_Months",
        F.when(col("Credit_History_Age").contains("Years"), 
            F.regexp_extract(col("Credit_History_Age"), "(\\d+) Years", 1).cast("int") * 12 + 
            F.when(col("Credit_History_Age").contains("Months"), F.regexp_extract(col("Credit_History_Age"), "and (\\d+) Months", 1).cast("int")).otherwise(0)
        ).otherwise(
            F.when(col("Credit_History_Age").contains("Months"), F.regexp_extract(col("Credit_History_Age"), "(\\d+) Months", 1).cast("int")).otherwise(0)
        )
    )
    
    # Set appropriate data types for other columns
    numeric_columns = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
    ]
    
    for col_name in numeric_columns:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
    
    # Feature engineering
    
    # Debt-to-Income ratio
    df = df.withColumn("Debt_to_Income", 
                      F.when(col("Annual_Income") > 0, 
                             col("Outstanding_Debt") / (col("Annual_Income") / 12)).otherwise(0))
    
    # Payment-to-Income ratio
    df = df.withColumn("Payment_to_Income", 
                      F.when(col("Monthly_Inhand_Salary") > 0, 
                             col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")).otherwise(0))
    
    # Income after obligations
    df = df.withColumn("Disposable_Income", 
                      col("Monthly_Inhand_Salary") - col("Total_EMI_per_month"))
    
    # Does minimum payment (binary)
    df = df.withColumn("Does_Min_Payment", 
                      F.when(col("Payment_of_Min_Amount") == "Yes", 1).otherwise(0))
    
    # Map Payment_Behaviour to risk scores
    df = df.withColumn("Payment_Behaviour_Score",
                     F.when(col("Payment_Behaviour").contains("High_spent"), 3)
                      .when(col("Payment_Behaviour").contains("Medium_spent"), 2)
                      .when(col("Payment_Behaviour").contains("Low_spent"), 1)
                      .otherwise(0))
    
    # Number of loan types
    df = df.withColumn("Num_Loan_Types", 
                      F.when(col("Type_of_Loan") == "None", 0)
                       .otherwise(F.size(F.split(col("Type_of_Loan"), ","))))
    
    # Credit Mix Score
    df = df.withColumn("Credit_Mix_Score",
                     F.when(col("Credit_Mix") == "Good", 2)
                      .when(col("Credit_Mix") == "Standard", 1)
                      .when(col("Credit_Mix") == "Bad", 0)
                      .otherwise(1))  # Unknown gets middle score
                      
    # save silver table 
    partition_name = f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet"
    filepath = os.path.join(silver_financials_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_silver_loan(snapshot_date_str, bronze_loan_directory, silver_loan_directory, spark):
    """
    Process loan daily data into silver layer with data cleaning and MOB calculation
    This follows the same logic as the working label store code
    """
     
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_loan_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table 
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df