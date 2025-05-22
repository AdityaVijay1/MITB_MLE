import os
import glob
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType


def process_gold_feature_store(snapshot_date_str, 
                               silver_clickstream_directory,
                               silver_attributes_directory,
                               silver_financials_directory,
                               silver_loan_directory,
                               gold_feature_store_directory,
                               spark):
    """
    Process all silver layer data into a unified gold feature store
    Following the criteria: filter loan data by mob=6 and join on customer_id
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Load silver tables
    clickstream_path = os.path.join(silver_clickstream_directory, f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    attributes_path = os.path.join(silver_attributes_directory, f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    financials_path = os.path.join(silver_financials_directory, f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet")
    loan_path = os.path.join(silver_loan_directory, f"silver_loan_daily_{snapshot_date_str.replace('-','_')}.parquet")
    
    # Read the data
    clickstream_df = spark.read.parquet(clickstream_path)
    attributes_df = spark.read.parquet(attributes_path)
    financials_df = spark.read.parquet(financials_path)
    loan_df = spark.read.parquet(loan_path)
    
    print(f'Loaded clickstream: {clickstream_df.count()} rows')
    print(f'Loaded attributes: {attributes_df.count()} rows')
    print(f'Loaded financials: {financials_df.count()} rows')
    print(f'Loaded loan: {loan_df.count()} rows')
    
    # First criteria: filter loan data by mob = 6
    loan_mob6_df = loan_df.filter(col("mob") == 6)
    print(f'Loan data filtered by mob=6: {loan_mob6_df.count()} rows')
    
    # Start with clickstream as base 
    feature_store_df = clickstream_df
    
    # Join with loan data (mob=6) on Customer_ID to get loan_id
    feature_store_df = feature_store_df.join(
        loan_mob6_df.select(
            "Customer_ID",
            "loan_id",
            "loan_amt",
            "balance",
            "overdue_amt",
            "paid_amt",
            "due_amt",
            "tenure",
            "installment_num",
            "mob",
            "dpd"
        ),
        on="Customer_ID",
        how="left"
    )
    
    # Join with attributes on Customer_ID (left join)
    feature_store_df = feature_store_df.join(
        attributes_df.select(
            "Customer_ID",
            "Age",
            "Age_Bucket",
            "is_Lawyer",
            "is_Mechanic",
            "is_MediaManager",
            "SSN_valid"
        ),
        on="Customer_ID",
        how="left"
    )
    
    # Join with financials on Customer_ID (left join)
    feature_store_df = feature_store_df.join(
        financials_df.select(
            "Customer_ID",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Credit_Mix_Score",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Credit_History_Months",
            "Does_Min_Payment",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Payment_Behaviour_Score",
            "Monthly_Balance",
            "Debt_to_Income",
            "Payment_to_Income",
            "Disposable_Income",
            "Num_Loan_Types"
        ),
        on="Customer_ID",
        how="left"
    )
    
    
    # Risk score combining credit and financial behaviors
    feature_store_df = feature_store_df.withColumn(
        "Combined_Risk_Score",
        F.coalesce(col("Credit_Mix_Score"), lit(0)) + 
        F.coalesce(col("Payment_Behaviour_Score"), lit(0)) - 
        F.when(F.coalesce(col("Num_of_Delayed_Payment"), lit(0)) > 0, 1).otherwise(0)
    )
    
    # Digital engagement score based on clickstream data
    feature_store_df = feature_store_df.withColumn(
        "Digital_Engagement_Score",
        F.when(F.coalesce(col("fe_sum"), lit(0)) > 0, 1).otherwise(0)
    )
    
    # Normalize important financial metrics by age group
    feature_store_df = feature_store_df.withColumn(
        "Income_to_Age_Ratio",
        F.coalesce(col("Annual_Income"), lit(0)) / F.when(F.coalesce(col("Age"), lit(0)) > 0, col("Age")).otherwise(30)
    )
    
    # Loan-specific features
    feature_store_df = feature_store_df.withColumn(
        "loan_utilization_ratio",
        F.when(F.coalesce(col("loan_amt"), lit(0)) > 0, 
               F.coalesce(col("balance"), lit(0)) / col("loan_amt")).otherwise(0)
    )
    
    feature_store_df = feature_store_df.withColumn(
        "payment_completion_ratio",
        F.when(F.coalesce(col("loan_amt"), lit(0)) > 0, 
               F.coalesce(col("paid_amt"), lit(0)) / col("loan_amt")).otherwise(0)
    )
    
    # Add version information and metadata
    feature_store_df = feature_store_df.withColumn("feature_store_version", lit("1.0"))
    feature_store_df = feature_store_df.withColumn("processing_date", F.to_timestamp(lit(snapshot_date)))
    
    # Ensure all numeric columns are filled with appropriate default values
    numeric_columns = [col_name for col_name, data_type in feature_store_df.dtypes 
                      if data_type in ['int', 'bigint', 'float', 'double'] and 
                      col_name not in ["Customer_ID", "snapshot_date", "processing_date"]]
    
    for column in numeric_columns:
        feature_store_df = feature_store_df.withColumn(
            column, 
            F.coalesce(col(column), lit(0.0).cast(DoubleType()))
        )
    
    # Final output should match number of Customer_IDs in the clickstream data
    print(f'Final feature store row count: {feature_store_df.count()}')
    print(f'Records with loan_id (mob=6): {feature_store_df.filter(col("loan_id").isNotNull()).count()}')
    
    # Save gold table
    partition_name = f"gold_feature_store_{snapshot_date_str.replace('-','_')}.parquet"
    filepath = os.path.join(gold_feature_store_directory, partition_name)
    feature_store_df.write.mode("overwrite").parquet(filepath)
    print('Saved to:', filepath)
    
    return feature_store_df