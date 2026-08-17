import sys
import boto3
import pandas as pd

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log(message):
    print(f"[ETL-VALIDATION] {message}")


def parse_s3_path(s3_path):

    path = s3_path.replace("s3://", "")

    bucket = path.split("/", 1)[0]
    key = path.split("/", 1)[1]

    return bucket, key


# ============================================================
# 1. READ PARAMETERS
# ============================================================

log("================================================")
log("STARTING VALIDATION / REVALIDATION GLUE JOB")
log("================================================")

args = getResolvedOptions(
    sys.argv,
    [
        "SOURCE_PATH",
        "TARGET_PATH",
        "REPORT_PATH",
        "RUN_TYPE",
        "SNS_TOPIC_ARN",
        "RECONCILIATION_REPORT_PATH"
    ]
)

source_path = args["SOURCE_PATH"]
target_path = args["TARGET_PATH"]
report_path = args["REPORT_PATH"]
run_type = args["RUN_TYPE"]
sns_topic_arn = args["SNS_TOPIC_ARN"]
reconciliation_report_path = args[
    "RECONCILIATION_REPORT_PATH"
]


log(f"RUN_TYPE = {run_type}")
log(f"SOURCE_PATH = {source_path}")
log(f"TARGET_PATH = {target_path}")
log(f"REPORT_PATH = {report_path}")


# ============================================================
# 2. INITIALIZE SPARK / GLUE / AWS CLIENTS
# ============================================================

log("[STEP 1] Initializing Spark, Glue and AWS clients")

sc = SparkContext.getOrCreate()

glueContext = GlueContext(sc)

spark = glueContext.spark_session

s3 = boto3.client("s3")

sns = boto3.client("sns")

log("[STEP 1] Initialization completed")


# ============================================================
# 3. PARSE REPORT PATH
# ============================================================

report_bucket, report_key = parse_s3_path(
    report_path
)

log(
    f"Report bucket = {report_bucket}, "
    f"Report key = {report_key}"
)


# ============================================================
# 4. READ SOURCE
# ============================================================

log("[STEP 2] Reading Source CSV")

try:

    source_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(source_path)
    )

    source_count = source_df.count()

    log(
        f"[STEP 2] Source successfully read. "
        f"Record count = {source_count}"
    )

except Exception as e:

    log(
        f"[ERROR] Failed to read Source: {str(e)}"
    )

    raise


# ============================================================
# 5. READ TARGET
# ============================================================

log("[STEP 3] Reading Target CSV")

try:

    target_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(target_path)
    )

    target_count = target_df.count()

    log(
        f"[STEP 3] Target successfully read. "
        f"Record count = {target_count}"
    )

except Exception as e:

    log(
        f"[ERROR] Failed to read Target: {str(e)}"
    )

    raise


# ============================================================
# 6. RECORD COUNT VALIDATION
# ============================================================

log("[STEP 4] Performing record count validation")

if source_count == target_count:

    count_status = "PASS"

else:

    count_status = "FAIL"


log(
    f"Source count = {source_count}"
)

log(
    f"Target count = {target_count}"
)

log(
    f"Record count validation = {count_status}"
)


# ============================================================
# 7. CUSTOMER ID COMPARISON
# ============================================================

log("[STEP 5] Comparing customer IDs")

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
    "[STEP 5] Finding records present "
    "in Source but missing from Target"
)

extra_in_source = (
    source_ids.subtract(target_ids)
)


log(
    "[STEP 5] Finding records present "
    "in Target but missing from Source"
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
    f"Extra records in Source = "
    f"{extra_in_source_count}"
)

log(
    f"Extra records in Target = "
    f"{extra_in_target_count}"
)


# ============================================================
# 8. OVERALL VALIDATION RESULT
# ============================================================

if (
    count_status == "PASS"
    and extra_in_source_count == 0
    and extra_in_target_count == 0
):

    validation_status = "PASSED"

else:

    validation_status = "FAILED"


log(
    f"[STEP 6] Overall {run_type} result = "
    f"{validation_status}"
)


# ============================================================
# 9. CREATE SUMMARY REPORT
# ============================================================

log("[STEP 7] Creating Excel report")

summary_rows = [

    {
        "validation_check": "Record Count",
        "source_value": source_count,
        "target_value": target_count,
        "status": count_status
    },

    {
        "validation_check": "Extra Records in Source",
        "source_value": extra_in_source_count,
        "target_value": "-",
        "status": (
            "PASS"
            if extra_in_source_count == 0
            else "FAIL"
        )
    },

    {
        "validation_check": "Extra Records in Target",
        "source_value": "-",
        "target_value": extra_in_target_count,
        "status": (
            "PASS"
            if extra_in_target_count == 0
            else "FAIL"
        )
    },

    {
        "validation_check": f"Overall {run_type}",
        "source_value": "-",
        "target_value": "-",
        "status": validation_status
    }
]


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# 10. CREATE MISMATCH DETAILS
# ============================================================

log("[STEP 8] Creating mismatch details")

mismatch_rows = []


for row in extra_in_source.collect():

    mismatch_rows.append(
        {
            "customer_id": row["customer_id"],
            "issue": "EXTRA_IN_SOURCE"
        }
    )


for row in extra_in_target.collect():

    mismatch_rows.append(
        {
            "customer_id": row["customer_id"],
            "issue": "EXTRA_IN_TARGET"
        }
    )


mismatch_df = pd.DataFrame(
    mismatch_rows,
    columns=[
        "customer_id",
        "issue"
    ]
)


# ============================================================
# 11. CREATE EXCEL FILE
# ============================================================

local_report_path = "/tmp/validation_report.xlsx"

log(
    f"[STEP 9] Creating Excel file "
    f"at {local_report_path}"
)

with pd.ExcelWriter(
    local_report_path,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Validation Summary",
        index=False
    )

    mismatch_df.to_excel(
        writer,
        sheet_name="Mismatches",
        index=False
    )


log("[STEP 9] Excel report created successfully")


# ============================================================
# 12. UPLOAD REPORT TO S3
# ============================================================

log("[STEP 10] Uploading report to S3")

try:

    s3.upload_file(
        local_report_path,
        report_bucket,
        report_key
    )

    log(
        f"[STEP 10] Report uploaded successfully:"
    )

    log(
        f"s3://{report_bucket}/{report_key}"
    )

except Exception as e:

    log(
        f"[ERROR] Failed to upload report: {str(e)}"
    )

    raise


# ============================================================
# 13. GENERATE PRESIGNED URL
# ============================================================

log("[STEP 11] Generating report download URL")

report_url = s3.generate_presigned_url(
    ClientMethod="get_object",

    Params={
        "Bucket": report_bucket,
        "Key": report_key
    },

    ExpiresIn=86400
)


log(
    "[STEP 11] Presigned URL generated"
)


# ============================================================
# 14. IF REVALIDATION, SEND ALL 3 REPORT URLs
# ============================================================

if run_type == "REVALIDATION":

    log(
        "[STEP 12] Revalidation completed"
    )

    # --------------------------------------------------------
    # Validation report URL
    # --------------------------------------------------------

    validation_report_bucket, validation_report_key = (
        parse_s3_path(
            "s3://"
            + report_bucket
            + "/reports/validation_report.xlsx"
        )
    )

    validation_url = (
        s3.generate_presigned_url(
            ClientMethod="get_object",

            Params={
                "Bucket": validation_report_bucket,
                "Key": validation_report_key
            },

            ExpiresIn=86400
        )
    )


    # --------------------------------------------------------
    # Reconciliation report URL
    # --------------------------------------------------------

    reconciliation_bucket, reconciliation_key = (
        parse_s3_path(
            reconciliation_report_path
        )
    )

    reconciliation_url = (
        s3.generate_presigned_url(
            ClientMethod="get_object",

            Params={
                "Bucket": reconciliation_bucket,
                "Key": reconciliation_key
            },

            ExpiresIn=86400
        )
    )


    # --------------------------------------------------------
    # Final SNS message
    # --------------------------------------------------------

    sns_message = f"""
ETL POST-MIGRATION PIPELINE COMPLETED

========================================
FINAL STATUS
========================================

Revalidation Status:
{validation_status}

Target has been revalidated against Source.

========================================
VALIDATION REPORT
========================================

Download:
{validation_url}

========================================
RECONCILIATION REPORT
========================================

Download:
{reconciliation_url}

========================================
REVALIDATION REPORT
========================================

Download:
{report_url}

========================================

All report links are valid for 24 hours.
"""


    log(
        "[STEP 12] Sending final SNS notification"
    )

    try:

        sns.publish(
            TopicArn=sns_topic_arn,

            Subject=(
                "ETL Pipeline - "
                "Validation, Reconciliation & Revalidation"
            ),

            Message=sns_message
        )

        log(
            "[STEP 12] Final SNS notification sent successfully"
        )

    except Exception as e:

        log(
            f"[ERROR] Failed to send SNS notification: "
            f"{str(e)}"
        )

        raise


# ============================================================
# 15. FINAL RESULT
# ============================================================

log("================================================")
log(
    f"{run_type} JOB COMPLETED"
)
log(
    f"FINAL STATUS = {validation_status}"
)
log("================================================")


# ============================================================
# 16. FAIL JOB IF VALIDATION FAILED
# ============================================================

if validation_status == "FAILED":

    log(
        "[ERROR] Validation/Revalidation failed."
    )

    raise Exception(
        f"{run_type} FAILED: "
        "Source and Target data do not match."
    )


log(
    f"{run_type} PASSED successfully."
)
