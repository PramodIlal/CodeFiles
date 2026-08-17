import sys
import boto3
import pandas as pd

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext


# ============================================================
# 1. GET PARAMETERS
# ============================================================

args = getResolvedOptions(
    sys.argv,
    [
        "SOURCE_PATH",
        "TARGET_PATH",
        "REPORT_PATH",
        "SNS_TOPIC_ARN"
    ]
)

source_path = args["SOURCE_PATH"]
target_path = args["TARGET_PATH"]
report_path = args["REPORT_PATH"]
sns_topic_arn = args["SNS_TOPIC_ARN"]


# ============================================================
# 2. INITIALIZE GLUE / SPARK / AWS CLIENTS
# ============================================================

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

s3 = boto3.client("s3")
sns = boto3.client("sns")


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def parse_s3_path(s3_path):

    s3_path = s3_path.replace("s3://", "")

    bucket = s3_path.split("/", 1)[0]
    key = s3_path.split("/", 1)[1]

    return bucket, key


# ============================================================
# 4. PARSE S3 PATHS
# ============================================================

source_bucket, source_key = parse_s3_path(source_path)

target_bucket, target_key = parse_s3_path(target_path)

report_bucket, report_key = parse_s3_path(report_path)


# ============================================================
# 5. READ SOURCE
# ============================================================

print("========================================")
print("READING SOURCE DATA")
print("========================================")

source_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(source_path)
)


# ============================================================
# 6. READ TARGET
# ============================================================

print("========================================")
print("READING TARGET DATA")
print("========================================")

target_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(target_path)
)


# ============================================================
# 7. FIND DIFFERENCES
# ============================================================

source_ids = (
    source_df
    .select("customer_id")
    .distinct()
)

target_ids = (
    target_df
    .select("customer_id")
    .distinct()
)


# Source records missing from Target
extra_in_source = source_ids.subtract(target_ids)

# Target records not present in Source
extra_in_target = target_ids.subtract(source_ids)


extra_in_source_count = extra_in_source.count()
extra_in_target_count = extra_in_target.count()


print("========================================")
print("RECONCILIATION RESULTS")
print("========================================")

print(
    f"Extra in Source : {extra_in_source_count}"
)

print(
    f"Extra in Target : {extra_in_target_count}"
)


# ============================================================
# 8. CREATE EXCEL REPORT DATA
# ============================================================

extra_source_pd = extra_in_source.toPandas()

extra_target_pd = extra_in_target.toPandas()


report_rows = []


# Records that need to be added to Target
for _, row in extra_source_pd.iterrows():

    report_rows.append(
        {
            "customer_id": row["customer_id"],
            "issue": "EXTRA_IN_SOURCE",
            "action_taken": "ADDED_TO_TARGET"
        }
    )


# Records that need to be removed from Target
for _, row in extra_target_pd.iterrows():

    report_rows.append(
        {
            "customer_id": row["customer_id"],
            "issue": "EXTRA_IN_TARGET",
            "action_taken": "REMOVED_FROM_TARGET"
        }
    )


if report_rows:

    reconciliation_report = pd.DataFrame(
        report_rows
    )

else:

    reconciliation_report = pd.DataFrame(
        columns=[
            "customer_id",
            "issue",
            "action_taken"
        ]
    )


print("========================================")
print("RECONCILIATION REPORT")
print("========================================")

print(
    reconciliation_report.to_string(index=False)
)


# ============================================================
# 9. CREATE EXCEL FILE
# ============================================================

local_report_path = (
    "/tmp/customer_reconciliation_report.xlsx"
)


with pd.ExcelWriter(
    local_report_path,
    engine="openpyxl"
) as writer:

    reconciliation_report.to_excel(
        writer,
        sheet_name="Reconciliation",
        index=False
    )


# ============================================================
# 10. UPLOAD EXCEL REPORT TO S3
# ============================================================

print("========================================")
print("UPLOADING EXCEL REPORT TO S3")
print("========================================")

s3.upload_file(
    local_report_path,
    report_bucket,
    report_key
)


report_s3_path = (
    "s3://"
    + report_bucket
    + "/"
    + report_key
)


print("Report created:")
print(report_s3_path)


# ============================================================
# 11. CREATE PRESIGNED DOWNLOAD URL
# ============================================================

download_url = s3.generate_presigned_url(
    ClientMethod="get_object",
    Params={
        "Bucket": report_bucket,
        "Key": report_key
    },
    ExpiresIn=86400
)


print("Presigned download URL generated.")


# ============================================================
# 12. SYNCHRONIZE TARGET WITH SOURCE
#
# Source is the master.
# ============================================================

print("========================================")
print("SYNCHRONIZING TARGET WITH SOURCE")
print("========================================")


# Delete existing target file
try:

    s3.delete_object(
        Bucket=target_bucket,
        Key=target_key
    )

    print("Existing target deleted.")

except Exception as e:

    print("Existing target could not be deleted:")
    print(e)


# Copy source file to target
s3.copy_object(
    Bucket=target_bucket,

    CopySource={
        "Bucket": source_bucket,
        "Key": source_key
    },

    Key=target_key
)


print("Target synchronized successfully.")


# ============================================================
# 13. SEND RECONCILIATION REPORT THROUGH SNS
# ============================================================

sns_message = f"""
Post-Migration Reconciliation Completed

Source: {source_path}
Target: {target_path}

Reconciliation Summary
----------------------
Extra in Source : {extra_in_source_count}
Extra in Target : {extra_in_target_count}

Target has been synchronized with Source.

Reconciliation Report
---------------------
Report: {report_s3_path}

Download Report:
{download_url}

The download link is valid for 24 hours.
"""


sns.publish(
    TopicArn=sns_topic_arn,
    Subject="ETL Reconciliation Report",
    Message=sns_message
)


print("========================================")
print("SNS NOTIFICATION SENT")
print("========================================")


# ============================================================
# 14. FINAL RESULT
# ============================================================

print("========================================")
print("RECONCILIATION COMPLETED")
print("========================================")

print("Target is now synchronized with Source.")

print(
    "Report:"
    + report_s3_path
)
