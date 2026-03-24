import sys
import json
import boto3
import pandas as pd
from io import BytesIO
from urllib.parse import urlparse
from datetime import datetime

from awsglue.utils import getResolvedOptions


# =========================================================
# S3 Helpers
# =========================================================

def parse_s3_uri(s3_uri: str):
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    return bucket, key


def read_text_from_s3(s3_client, s3_uri: str) -> str:
    bucket, key = parse_s3_uri(s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8").strip()


def read_excel_from_s3(s3_client, s3_uri: str) -> pd.ExcelFile:
    bucket, key = parse_s3_uri(s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.ExcelFile(BytesIO(obj["Body"].read()))


def upload_bytes_to_s3(s3_client, data_bytes: bytes, s3_uri: str, content_type: str = None):
    bucket, key = parse_s3_uri(s3_uri)
    kwargs = {"Bucket": bucket, "Key": key, "Body": data_bytes}
    if content_type:
        kwargs["ContentType"] = content_type
    s3_client.put_object(**kwargs)


# =========================================================
# Utility
# =========================================================

def safe_get(row, col_name, default=None):
    try:
        value = row[col_name]
        return default if pd.isna(value) else value
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def delta(src, tgt):
    return to_int(src) - to_int(tgt)


def resolve_report_path_from_latest_txt(s3_client, path, suffix=".xlsx"):
    resolved = read_text_from_s3(s3_client, path)
    if not resolved.startswith("s3://") or not resolved.lower().endswith(suffix):
        raise ValueError(f"Invalid resolved path: {resolved}")
    return resolved


# =========================================================
# Main
# =========================================================

def main():
    args = getResolvedOptions(sys.argv, [
        "VALIDATION_LATEST_TXT_PATH",
        "RECON_LATEST_TXT_PATH",
        "REVALIDATION_LATEST_TXT_PATH",
        "OUTPUT_BASE_PATH",
        "executionArn",
        "mappingRunId",
        "validationRunId",
        "reconciliationRunId",
        "revalidationRunId"
    ])

    # Inputs
    executionArn = args["executionArn"]
    mapping_run_id = args["mappingRunId"]
    validation_run_id = args["validationRunId"]
    recon_run_id = args["reconciliationRunId"]
    reval_run_id = args["revalidationRunId"]

    validation_latest_txt_path = args["VALIDATION_LATEST_TXT_PATH"]
    recon_latest_txt_path = args["RECON_LATEST_TXT_PATH"]
    revalidation_latest_txt_path = args["REVALIDATION_LATEST_TXT_PATH"]
    output_base_path = args["OUTPUT_BASE_PATH"].rstrip("/")

    # Clients
    s3 = boto3.client("s3")
    sf = boto3.client("stepfunctions")
    glue = boto3.client("glue")

    # =====================================================
    # Execution Time
    # =====================================================
    exec_resp = sf.describe_execution(executionArn=executionArn)
    start_time = exec_resp["startDate"]
    end_time = datetime.utcnow()
    duration_seconds = (end_time - start_time).total_seconds()

    # =====================================================
    # Glue Cost
    # =====================================================
    GLUE_COST_PER_DPU_HOUR = 0.44

    def get_glue_cost(job_name, run_id):
        try:
            res = glue.get_job_run(JobName=job_name, RunId=run_id)
            dpu = res["JobRun"].get("DPUSeconds", 0)
            return dpu, (dpu / 3600) * GLUE_COST_PER_DPU_HOUR
        except Exception as e:
            print(f"Cost fetch failed for {job_name}: {e}")
            return 0, 0

    mapping_dpu, mapping_cost = get_glue_cost("Mapping_Generator", mapping_run_id)
    validation_dpu, validation_cost = get_glue_cost("Glue_Validation_Job", validation_run_id)
    recon_dpu, recon_cost = get_glue_cost("Glue_Reconciliation_Job", recon_run_id)
    reval_dpu, reval_cost = get_glue_cost("Glue_Validation_Job", reval_run_id)

    total_glue_cost = mapping_cost + validation_cost + recon_cost + reval_cost

    # Step Function + SNS (approx)
    sf_cost = 8 * 0.000025
    sns_cost = 0.0000005

    total_cost = total_glue_cost + sf_cost + sns_cost

    # =====================================================
    # Load Reports
    # =====================================================
    val_path = resolve_report_path_from_latest_txt(s3, validation_latest_txt_path)
    recon_path = resolve_report_path_from_latest_txt(s3, recon_latest_txt_path)
    reval_path = resolve_report_path_from_latest_txt(s3, revalidation_latest_txt_path)

    val_excel = read_excel_from_s3(s3, val_path)
    recon_excel = read_excel_from_s3(s3, recon_path)
    reval_excel = read_excel_from_s3(s3, reval_path)

    val_sum = val_excel.parse("summary_report").iloc[0]
    recon_sum = recon_excel.parse("summary").iloc[0]
    reval_sum = reval_excel.parse("summary_report").iloc[0]

    # =====================================================
    # Business Summary
    # =====================================================
    business_summary = pd.DataFrame([{
        "Check": "Row Count",
        "Before": delta(safe_get(val_sum, "source_count"), safe_get(val_sum, "target_count")),
        "After": delta(safe_get(reval_sum, "source_count"), safe_get(reval_sum, "target_count"))
    }])

    # =====================================================
    # Cost Summary (MERGED)
    # =====================================================
    cost_summary = pd.DataFrame([{
        "ExecutionArn": executionArn,
        "StartTime": str(start_time),
        "EndTime": str(end_time),
        "DurationSec": duration_seconds,
        "GlueCost": total_glue_cost,
        "StepFunctionCost": sf_cost,
        "SNSCost": sns_cost,
        "TotalCost": total_cost
    }])

    # =====================================================
    # Write Excel
    # =====================================================
    run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_base_path}/run_{run_ts}/closure_report.xlsx"

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        business_summary.to_excel(writer, "Business", index=False)
        cost_summary.to_excel(writer, "Cost_Summary", index=False)

    buffer.seek(0)

    upload_bytes_to_s3(s3, buffer.getvalue(), output_path,
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # =====================================================
    # Metadata
    # =====================================================
    metadata = {
        "execution": {
            "executionArn": executionArn,
            "start": str(start_time),
            "end": str(end_time),
            "duration": duration_seconds
        },
        "cost": {
            "glue": total_glue_cost,
            "step_function": sf_cost,
            "sns": sns_cost,
            "total": total_cost
        },
        "output": output_path
    }

    upload_bytes_to_s3(
        s3,
        json.dumps(metadata, indent=2).encode(),
        output_path.replace(".xlsx", ".json"),
        "application/json"
    )

    print("SUCCESS")


if __name__ == "__main__":
    main()
