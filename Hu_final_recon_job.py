import sys
import boto3
import pandas as pd

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext


# ============================================================
# HELPER
# ============================================================

def log(message):
    print(f"[ETL-RECONCILIATION] {message}")


def parse_s3_path(s3_path):

    path = s3_path.replace("s3://", "")

    bucket = path.split("/", 1)[0]
    key = path.split("/", 1)[1]

    return bucket, key


# ============================================================
# 1. START
# ============================================================

log("================================================")
log("STARTING RECONCILIATION GLUE JOB")
log("================================================")


# ============================================================
# 2. PARAMETERS
# ============================================================

args = getResolvedOptions(
    sys.argv,
    [
        "SOURCE_PATH",
        "TARGET_PATH",
        "REPORT_PATH"
    ]
)

source_path = args["SOURCE_PATH"]
target_path = args["TARGET_PATH"]
report_path = args["REPORT_PATH"]


log(f"SOURCE_PATH = {source_path}")
log(f"TARGET_PATH = {target_path}")
log(f"REPORT_PATH = {report_path}")


# ============================================================
# 3. INITIALIZE
# ============================================================

log("[STEP 1] Initializing Spark and AWS clients")

sc = SparkContext.getOrCreate()

glueContext = GlueContext(sc)

spark = glueContext.spark_session

s3 = boto3.client("s3")

log("[STEP 1] Initialization completed")


# ============================================================
# 4. PARSE S3 PATHS
# ============================================================

source_bucket, source_key = (
    parse_s3_path(source_path)
)

target_bucket, target_key = (
    parse_s3_path(target_path)
)

report_bucket, report_key = (
    parse_s3_path(report_path)
)


# ============================================================
# 5. READ SOURCE
# ============================================================

log("[STEP 2] Reading Source data")

try:

    source_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(source_path)
    )

    source_count = source_df.count()

    log(
        f"[STEP 2] Source record count = "
        f"{source_count}"
    )

except Exception as e:

    log(
        f"[ERROR] Failed to read Source: {str(e)}"
    )

    raise


# ============================================================
# 6. READ TARGET
# ============================================================

log("[STEP 3] Reading Target data")

try:

    target_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(target_path)
    )

    target_count = target_df.count()

    log(
        f"[STEP 3] Target record count = "
        f"{target_count}"
    )

except Exception as e:

    log(
        f"[ERROR] Failed to read Target: {str(e)}"
    )

    raise


# ============================================================
# 7. COMPARE DATA
# ============================================================

log("[STEP 4] Comparing Source and Target")


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


log(
    "[STEP 4.1] Finding records "
    "missing from Target"
)

extra_in_source = (
    source_ids.subtract(target_ids)
)


log(
    "[STEP 4.2] Finding records "
    "extra in Target"
)

extra_in_target = (
    target_ids.subtract(source_ids)
)


extra_in_source_count = (
    extra_in_source.count()
)

extra_in_target_count = (
    extra_in_target.count()
)


log(
    f"Extra in Source = "
    f"{extra_in_source_count}"
)

log(
    f"Extra in Target = "
    f"{extra_in_target_count}"
)


# ============================================================
# 8. SHOW DIFFERENCES IN CLOUDWATCH
# ============================================================

if extra_in_source_count > 0:

    log(
        "[STEP 5] Records EXTRA IN SOURCE:"
    )

    extra_in_source.show(
        truncate=False
    )

else:

    log(
        "[STEP 5] No records extra in Source"
    )


if extra_in_target_count > 0:

    log(
        "[STEP 5] Records EXTRA IN TARGET:"
    )

    extra_in_target.show(
        truncate=False
    )

else:

    log(
        "[STEP 5] No records extra in Target"
    )


# ============================================================
# 9. CREATE RECONCILIATION REPORT
# ============================================================

log(
    "[STEP 6] Creating reconciliation report"
)

report_rows = []


# Source → Target
for row in extra_in_source.collect():

    report_rows.append(
        {
            "customer_id": row["customer_id"],
            "issue": "EXTRA_IN_SOURCE",
            "action_taken": "ADDED_TO_TARGET"
        }
    )


# Target → Source
for row in extra_in_target.collect():

    report_rows.append(
        {
            "customer_id": row["customer_id"],
            "issue": "EXTRA_IN_TARGET",
            "action_taken": "REMOVED_FROM_TARGET"
        }
    )


reconciliation_df = pd.DataFrame(
    report_rows,

    columns=[
        "customer_id",
        "issue",
        "action_taken"
    ]
)


# ============================================================
# 10. CREATE EXCEL
# ============================================================

local_report_path = (
    "/tmp/reconciliation_report.xlsx"
)

log(
    "[STEP 7] Creating Excel file"
)

with pd.ExcelWriter(
    local_report_path,
    engine="openpyxl"
) as writer:

    reconciliation_df.to_excel(
        writer,
        sheet_name="Reconciliation",
        index=False
    )


log(
    "[STEP 7] Excel report created successfully"
)


# ============================================================
# 11. UPLOAD REPORT
# ============================================================

log(
    "[STEP 8] Uploading reconciliation report to S3"
)

try:

    s3.upload_file(
        local_report_path,
        report_bucket,
        report_key
    )

    log(
        f"[STEP 8] Report uploaded:"
    )

    log(
        f"s3://{report_bucket}/{report_key}"
    )

except Exception as e:

    log(
        f"[ERROR] Report upload failed: {str(e)}"
    )

    raise


# ============================================================
# 12. SYNCHRONIZE TARGET
# ============================================================

log(
    "[STEP 9] Starting Target synchronization"
)

log(
    "Source is the MASTER dataset."
)

log(
    "Target will be replaced with Source."
)


# Delete old Target

try:

    s3.delete_object(
        Bucket=target_bucket,
        Key=target_key
    )

    log(
        "[STEP 9.1] Existing Target deleted"
    )

except Exception as e:

    log(
        f"[WARNING] Could not delete existing Target: "
        f"{str(e)}"
    )


# Copy Source → Target

try:

    s3.copy_object(
        Bucket=target_bucket,

        CopySource={
            "Bucket": source_bucket,
            "Key": source_key
        },

        Key=target_key
    )

    log(
        "[STEP 9.2] Source successfully copied to Target"
    )

except Exception as e:

    log(
        f"[ERROR] Failed to synchronize Target: "
        f"{str(e)}"
    )

    raise


# ============================================================
# 13. FINAL
# ============================================================

log("================================================")
log("RECONCILIATION COMPLETED SUCCESSFULLY")
log("================================================")

log(
    f"Final Target = s3://{target_bucket}/{target_key}"
)

log(
    f"Reconciliation Report = "
    f"s3://{report_bucket}/{report_key}"
)

log(
    "Target is now synchronized with Source."
)

log("================================================")
