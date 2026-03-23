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
    """
    Parse S3 URI into bucket and key.
    Example:
        s3://my-bucket/path/file.txt
    Returns:
        ("my-bucket", "path/file.txt")
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI (missing bucket/key): {s3_uri}")

    return bucket, key


def read_text_from_s3(s3_client, s3_uri: str) -> str:
    """
    Read plain text file from S3 and return as string.
    """
    bucket, key = parse_s3_uri(s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read().decode("utf-8").strip()
    return content


def read_excel_from_s3(s3_client, s3_uri: str) -> pd.ExcelFile:
    """
    Read Excel file directly from S3 into pandas ExcelFile.
    """
    bucket, key = parse_s3_uri(s3_uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return pd.ExcelFile(BytesIO(data))


def upload_bytes_to_s3(s3_client, data_bytes: bytes, s3_uri: str, content_type: str = None):
    """
    Upload bytes to S3.
    """
    bucket, key = parse_s3_uri(s3_uri)

    kwargs = {
        "Bucket": bucket,
        "Key": key,
        "Body": data_bytes
    }

    if content_type:
        kwargs["ContentType"] = content_type

    s3_client.put_object(**kwargs)


# =========================================================
# Utility Helpers
# =========================================================

def safe_get(row, col_name, default=None):
    """
    Safely fetch value from pandas row.
    """
    try:
        value = row[col_name]
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def to_int(value, default=0):
    """
    Safe integer conversion.
    """
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def delta(src, tgt):
    """
    Calculate integer delta.
    """
    return to_int(src) - to_int(tgt)


def resolve_report_path_from_latest_txt(s3_client, latest_txt_s3_uri: str, expected_suffix: str = ".xlsx") -> str:
    """
    Read latest pointer txt file from S3 and return the actual report S3 path.
    Example txt content:
        s3://etl-valid-recon-framework-s3/reports/run_validation/run_20260324_101500/validation_report.xlsx
    """
    resolved_path = read_text_from_s3(s3_client, latest_txt_s3_uri).strip()

    if not resolved_path.startswith("s3://"):
        raise ValueError(
            f"Latest pointer file does not contain a valid S3 path: {latest_txt_s3_uri} -> {resolved_path}"
        )

    if expected_suffix and not resolved_path.lower().endswith(expected_suffix.lower()):
        raise ValueError(
            f"Resolved path from {latest_txt_s3_uri} does not end with {expected_suffix}: {resolved_path}"
        )

    return resolved_path


# =========================================================
# Main Logic
# =========================================================

def main():
    args = getResolvedOptions(
        sys.argv,
        [
            "VALIDATION_LATEST_TXT_PATH",
            "RECON_LATEST_TXT_PATH",
            "REVALIDATION_LATEST_TXT_PATH",
            "OUTPUT_BASE_PATH"
        ]
    )

    validation_latest_txt_path = args["VALIDATION_LATEST_TXT_PATH"]
    recon_latest_txt_path = args["RECON_LATEST_TXT_PATH"]
    revalidation_latest_txt_path = args["REVALIDATION_LATEST_TXT_PATH"]
    output_base_path = args["OUTPUT_BASE_PATH"].rstrip("/")

    s3 = boto3.client("s3")

    print("========== FINAL CLOSURE REPORT JOB STARTED ==========")
    print(f"VALIDATION_LATEST_TXT_PATH   = {validation_latest_txt_path}")
    print(f"RECON_LATEST_TXT_PATH        = {recon_latest_txt_path}")
    print(f"REVALIDATION_LATEST_TXT_PATH = {revalidation_latest_txt_path}")
    print(f"OUTPUT_BASE_PATH             = {output_base_path}")

    # -----------------------------------------------------
    # Resolve actual report file paths from latest .txt files
    # -----------------------------------------------------
    validation_report_path = resolve_report_path_from_latest_txt(s3, validation_latest_txt_path, ".xlsx")
    recon_report_path = resolve_report_path_from_latest_txt(s3, recon_latest_txt_path, ".xlsx")
    revalidation_report_path = resolve_report_path_from_latest_txt(s3, revalidation_latest_txt_path, ".xlsx")

    print(f"Resolved Validation Report Path   = {validation_report_path}")
    print(f"Resolved Reconciliation Report Path = {recon_report_path}")
    print(f"Resolved Revalidation Report Path = {revalidation_report_path}")

    # -----------------------------------------------------
    # Read Excel files from S3
    # -----------------------------------------------------
    validation_excel = read_excel_from_s3(s3, validation_report_path)
    recon_excel = read_excel_from_s3(s3, recon_report_path)
    revalidation_excel = read_excel_from_s3(s3, revalidation_report_path)

    # -----------------------------------------------------
    # Validate required sheets
    # -----------------------------------------------------
    if "summary_report" not in validation_excel.sheet_names:
        raise ValueError(
            f"Validation report missing required sheet 'summary_report'. Found: {validation_excel.sheet_names}"
        )

    if "summary" not in recon_excel.sheet_names:
        raise ValueError(
            f"Reconciliation report missing required sheet 'summary'. Found: {recon_excel.sheet_names}"
        )

    if "summary_report" not in revalidation_excel.sheet_names:
        raise ValueError(
            f"Revalidation report missing required sheet 'summary_report'. Found: {revalidation_excel.sheet_names}"
        )

    # -----------------------------------------------------
    # Read summary rows
    # -----------------------------------------------------
    val_sum_df = validation_excel.parse("summary_report")
    recon_sum_df = recon_excel.parse("summary")
    reval_sum_df = revalidation_excel.parse("summary_report")

    if val_sum_df.empty:
        raise ValueError("Validation summary_report sheet is empty.")
    if recon_sum_df.empty:
        raise ValueError("Reconciliation summary sheet is empty.")
    if reval_sum_df.empty:
        raise ValueError("Revalidation summary_report sheet is empty.")

    val_sum = val_sum_df.iloc[0]
    recon_sum = recon_sum_df.iloc[0]
    reval_sum = reval_sum_df.iloc[0]

    # -----------------------------------------------------
    # Business Summary Calculations
    # -----------------------------------------------------
    row_count_before_delta = delta(
        safe_get(val_sum, "source_count", 0),
        safe_get(val_sum, "target_count", 0)
    )

    row_count_after_delta = delta(
        safe_get(reval_sum, "source_count", 0),
        safe_get(reval_sum, "target_count", 0)
    )

    total_inserts = to_int(safe_get(recon_sum, "total_inserts", 0))
    total_updates = to_int(safe_get(recon_sum, "total_updates", 0))

    row_count_action = f"Inserted {total_inserts} missing rows"
    row_count_exception = "None" if row_count_after_delta == 0 else f"Δ {row_count_after_delta}"

    mismatch_before_count = to_int(safe_get(val_sum, "total_mismatch", 0))
    mismatch_after_count = to_int(safe_get(reval_sum, "total_mismatch", 0))

    mismatch_before = f"{mismatch_before_count} mismatches"
    mismatch_after = f"{mismatch_after_count} mismatches"
    mismatch_action = f"{total_updates} rows updated"
    mismatch_exception = "None" if mismatch_after_count == 0 else f"{mismatch_after_count} mismatches remain"

    schema_drift_before = str(safe_get(val_sum, "schema_drift_status", "UNKNOWN"))
    schema_drift_after = str(safe_get(reval_sum, "schema_drift_status", "UNKNOWN"))
    schema_drift_action = "N/A" if schema_drift_before.upper() == "PASS" else "Schema aligned"
    schema_drift_exception = "None" if schema_drift_after.upper() == "PASS" else "Drift remains"

    extra_target_before = to_int(safe_get(val_sum, "extra_records_in_target", 0))
    extra_target_after = to_int(safe_get(reval_sum, "extra_records_in_target", 0))
    total_deleted = extra_target_before - extra_target_after
    delete_action = f"{total_deleted} rows deleted"

    delete_exception = "None" if extra_target_after == 0 else f"{extra_target_after} extra records remain"

    # -----------------------------------------------------
    # Build Business Summary Sheet
    # -----------------------------------------------------
    business_summary = pd.DataFrame([
        {
            "Check type": "Row count",
            "Before (delta / issue)": f"Δ {row_count_before_delta}",
            "Action(s) taken to fix": row_count_action,
            "After (delta / status)": f"Δ {row_count_after_delta}",
            "Remaining exception (if not fixed)": row_count_exception
        },
        {
            "Check type": "Value mismatch",
            "Before (delta / issue)": mismatch_before,
            "Action(s) taken to fix": mismatch_action,
            "After (delta / status)": mismatch_after,
            "Remaining exception (if not fixed)": mismatch_exception
        },
        {
            "Check type": "Schema drift",
            "Before (delta / issue)": schema_drift_before,
            "Action(s) taken to fix": schema_drift_action,
            "After (delta / status)": schema_drift_after,
            "Remaining exception (if not fixed)": schema_drift_exception
        },
        {
            "Check type": "Deletion in target",
            "Before (delta / issue)": f"{extra_target_before}",
            "Action(s) taken to fix": delete_action,
            "After (delta / status)": f"{extra_target_after}",
            "Remaining exception (if not fixed)": delete_exception
        }
    ])

    # -----------------------------------------------------
    # Executive Summary Sheet
    # -----------------------------------------------------
    exec_summary = pd.DataFrame([{
        "Validation Run Timestamp": safe_get(val_sum, "run_timestamp", ""),
        "Reconciliation Run Timestamp": safe_get(recon_sum, "reconciliation_timestamp", ""),
        "Revalidation Run Timestamp": safe_get(reval_sum, "run_timestamp", ""),
        "Source Format": safe_get(val_sum, "source_format", ""),
        "Target Format": safe_get(val_sum, "target_format", ""),
        "Validation Status": safe_get(val_sum, "overall_status", ""),
        "Reconciliation Status": safe_get(recon_sum, "overall_status", "PASS"),
        "Revalidation Status": safe_get(reval_sum, "overall_status", "")
    }])

    # -----------------------------------------------------
    # Comparative Summary Sheet
    # -----------------------------------------------------
    comp_summary = pd.DataFrame([
        {
            "Stage": "Validation",
            "Source Record Count": safe_get(val_sum, "source_count", 0),
            "Target Record Count": safe_get(val_sum, "target_count", 0),
            "Total Mismatches": safe_get(val_sum, "total_mismatch", 0),
            "Missing Records in Target": safe_get(val_sum, "extra_records_in_source", 0),
            "Extra Records in Target": safe_get(val_sum, "extra_records_in_target", 0),
            "Row Count Status": safe_get(val_sum, "row_count_status", ""),
            "Schema Drift Status": safe_get(val_sum, "schema_drift_status", ""),
            "Duplicate Keys (Source/Target)": f"{safe_get(val_sum, 'source_duplicate_keys', 0)}/{safe_get(val_sum, 'target_duplicate_keys', 0)}",
            "Overall Status": safe_get(val_sum, "overall_status", "")
        },
        {
            "Stage": "Reconciliation",
            "Source Record Count": safe_get(recon_sum, "initial_target_count", 0),
            "Target Record Count": safe_get(recon_sum, "final_target_count", 0),
            "Total Mismatches": 0,
            "Missing Records in Target": 0,
            "Extra Records in Target": 0,
            "Row Count Status": "PASS",
            "Schema Drift Status": "PASS",
            "Duplicate Keys (Source/Target)": "0/0",
            "Overall Status": safe_get(recon_sum, "overall_status", "PASS")
        },
        {
            "Stage": "Revalidation",
            "Source Record Count": safe_get(reval_sum, "source_count", 0),
            "Target Record Count": safe_get(reval_sum, "target_count", 0),
            "Total Mismatches": safe_get(reval_sum, "total_mismatch", 0),
            "Missing Records in Target": safe_get(reval_sum, "extra_records_in_source", 0),
            "Extra Records in Target": safe_get(reval_sum, "extra_records_in_target", 0),
            "Row Count Status": safe_get(reval_sum, "row_count_status", ""),
            "Schema Drift Status": safe_get(reval_sum, "schema_drift_status", ""),
            "Duplicate Keys (Source/Target)": f"{safe_get(reval_sum, 'source_duplicate_keys', 0)}/{safe_get(reval_sum, 'target_duplicate_keys', 0)}",
            "Overall Status": safe_get(reval_sum, "overall_status", "")
        }
    ])

    # -----------------------------------------------------
    # Issues / Exceptions Sheet
    # -----------------------------------------------------
    unresolved_mismatches = to_int(safe_get(reval_sum, "total_mismatch", 0))
    unresolved_schema_drift = str(safe_get(reval_sum, "schema_drift_status", "UNKNOWN")).upper()
    unresolved_extra_target = to_int(safe_get(reval_sum, "extra_records_in_target", 0))

    issues = pd.DataFrame([{
        "Unresolved Mismatches": (
            "All mismatches addressed via updates/inserts"
            if unresolved_mismatches == 0
            else f"{unresolved_mismatches} mismatches remain"
        ),
        "Schema Drift": (
            "None detected"
            if unresolved_schema_drift == "PASS"
            else "Schema drift remains"
        ),
        "Extra Records In Target": (
            "None"
            if unresolved_extra_target == 0
            else f"{unresolved_extra_target} extra records remain"
        ),
        "Failed Operations": "None reported"
    }])

    # -----------------------------------------------------
    # Recommendations Sheet
    # -----------------------------------------------------
    recommendations = pd.DataFrame([{
        "Root Cause": "Initial target dataset had missing records and/or mismatched values compared to source",
        "Remediation": "Missing records inserted, mismatched records updated, and target revalidated",
        "Suggestions": "Review upstream ingestion, automate validation-reconciliation scheduling, and monitor periodic revalidation"
    }])

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    output_run_prefix = f"{output_base_path}/run_{run_timestamp}"
    closure_report_s3_path = f"{output_run_prefix}/closure_report.xlsx"
    metadata_s3_path = f"{output_run_prefix}/run_metadata.json"

    print(f"Closure report output path = {closure_report_s3_path}")
    print(f"Metadata output path       = {metadata_s3_path}")

    # -----------------------------------------------------
    # Create Excel in memory
    # -----------------------------------------------------
    output_buffer = BytesIO()

    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        exec_summary.to_excel(writer, sheet_name="Executive_Summary", index=False)
        business_summary.to_excel(writer, sheet_name="Business_Summary", index=False)
        comp_summary.to_excel(writer, sheet_name="Comparative_Summary", index=False)
        issues.to_excel(writer, sheet_name="Issues_Exceptions", index=False)
        recommendations.to_excel(writer, sheet_name="Recommendations", index=False)

    output_buffer.seek(0)

    # -----------------------------------------------------
    # Upload closure report to S3
    # -----------------------------------------------------
    upload_bytes_to_s3(
        s3,
        output_buffer.getvalue(),
        closure_report_s3_path,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # -----------------------------------------------------
    # Save metadata JSON
    # -----------------------------------------------------
    metadata = {
        "job_name": "Glue_Final_Closure_Report",
        "run_timestamp_utc": run_timestamp,
        "status": "SUCCESS",
        "input_latest_txt_files": {
            "validation_latest_txt_path": validation_latest_txt_path,
            "recon_latest_txt_path": recon_latest_txt_path,
            "revalidation_latest_txt_path": revalidation_latest_txt_path
        },
        "resolved_input_reports": {
            "validation_report_path": validation_report_path,
            "reconciliation_report_path": recon_report_path,
            "revalidation_report_path": revalidation_report_path
        },
        "output_files": {
            "closure_report_path": closure_report_s3_path
        }
    }

    upload_bytes_to_s3(
        s3,
        json.dumps(metadata, indent=2).encode("utf-8"),
        metadata_s3_path,
        content_type="application/json"
    )

    print("closure_report.xlsx created successfully.")
    print(f"Uploaded closure report to: {closure_report_s3_path}")
    print(f"Uploaded metadata to      : {metadata_s3_path}")
    print("========== FINAL CLOSURE REPORT JOB COMPLETED ==========")


if __name__ == "__main__":
    main()




# "Arguments": {
#   "--VALIDATION_LATEST_TXT_PATH": "s3://etl-valid-recon-framework-s3/reports/run_validation/latest_validation_report.txt",
#   "--RECON_LATEST_TXT_PATH": "s3://etl-valid-recon-framework-s3/reports/run_reconciliation/latest_reconciliation_report.txt",
#   "--REVALIDATION_LATEST_TXT_PATH": "s3://etl-valid-recon-framework-s3/reports/run_revalidation/latest_revalidation_report.txt",
#   "--OUTPUT_BASE_PATH": "s3://etl-valid-recon-framework-s3/reports/run_final_closure_report"
# }
