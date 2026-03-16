import json
import os
from datetime import datetime
from urllib.parse import urlparse

import boto3
import pandas as pd
from pyspark.sql import functions as F

# AWS Glue imports
import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext


# ============================================================
# GLUE JOB ARGUMENTS
# ============================================================
args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "SOURCE_PATH",
        "TARGET_PATH",
        "MAPPING_PATH",
        "REPORT_BASE_PATH",
        "RUN_ID"
    ]
)

SOURCE_PATH = args["SOURCE_PATH"]
TARGET_PATH = args["TARGET_PATH"]
MAPPING_PATH = args["MAPPING_PATH"]
REPORT_BASE_PATH = args["REPORT_BASE_PATH"]
RUN_ID = args["RUN_ID"]


# ============================================================
# S3 HELPERS
# ============================================================
def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def upload_file_to_s3(local_file_path, s3_uri):
    s3 = boto3.client("s3")
    bucket, key = parse_s3_uri(s3_uri)
    s3.upload_file(local_file_path, bucket, key)
    print(f"[OK] Uploaded to S3: {s3_uri}")


def read_text_from_s3(s3_uri):
    s3 = boto3.client("s3")
    bucket, key = parse_s3_uri(s3_uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def list_s3_objects(bucket, prefix):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def delete_s3_prefix(s3_uri):
    """
    Deletes all objects under an S3 prefix.
    Useful before overwrite to avoid stale files.
    """
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    keys = list_s3_objects(bucket, prefix)

    if not keys:
        return

    # delete in batches of 1000
    for i in range(0, len(keys), 1000):
        batch = keys[i:i+1000]
        delete_payload = {"Objects": [{"Key": k} for k in batch]}
        s3.delete_objects(Bucket=bucket, Delete=delete_payload)

    print(f"[OK] Cleared S3 prefix: {s3_uri}")


# ============================================================
# SPARK SESSION (GLUE COMPATIBLE)
# ============================================================
def create_spark_session(app_name="ETL Validation Job"):
    sc = SparkContext.getOrCreate()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    spark.sparkContext.setLogLevel("ERROR")
    return spark, glueContext


# ============================================================
# FILE READERS
# ============================================================
def read_csv_file(spark, file_path):
    """
    Reads a CSV file with header and inferSchema enabled.
    Works for S3 paths in Glue.
    """
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(file_path)
    )


def read_mapping_json(mapping_path):
    """
    Reads mapping JSON from S3.

    Expected format:
    {
      "primary_keys": [
        {"source_column": "...", "target_column": "..."}
      ],
      "column_mappings": [
        {"source_column": "...", "target_column": "..."}
      ]
    }
    """
    content = read_text_from_s3(mapping_path)
    mapping = json.loads(content)

    primary_keys = mapping.get("primary_keys", [])
    column_mappings = mapping.get("column_mappings", [])

    if not primary_keys:
        raise ValueError("Mapping JSON must contain 'primary_keys'.")

    if not column_mappings:
        raise ValueError("Mapping JSON must contain 'column_mappings'.")

    return primary_keys, column_mappings


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def print_section(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def create_run_output_dir(base_output_dir=REPORT_BASE_PATH, run_id=RUN_ID):
    """
    Creates run folder path in S3:
      s3://bucket/reports/validation/run_id=<RUN_ID>/
    """
    if not base_output_dir.endswith("/"):
        base_output_dir += "/"

    run_output_dir = f"{base_output_dir}run_id={run_id}/"
    return run_output_dir


def get_local_tmp_dir(run_id=RUN_ID):
    """
    Local temp folder inside Glue container for generating files before upload.
    """
    local_tmp_dir = f"/tmp/validation_run_{run_id}"
    os.makedirs(local_tmp_dir, exist_ok=True)
    return local_tmp_dir


def validate_required_columns(df, required_columns, dataset_name):
    """
    Checks whether all required columns exist in the dataframe.
    Returns list of missing columns.
    """
    df_columns = set(df.columns)
    missing_columns = [col for col in required_columns if col not in df_columns]

    if missing_columns:
        print(f"[ERROR] Missing columns in {dataset_name}: {missing_columns}")
    else:
        print(f"[OK] All required columns are present in {dataset_name}")

    return missing_columns


def rename_source_columns_to_common_names(source_df, column_mappings):
    """
    Renames source columns to target/common names based on mapping.
    Example:
      source.first_nm -> first_name
    """
    df = source_df
    for mapping in column_mappings:
        source_col = mapping["source_column"]
        target_col = mapping["target_column"]

        if source_col != target_col:
            df = df.withColumnRenamed(source_col, target_col)

    return df


def get_common_primary_keys(primary_keys):
    """
    After source columns are renamed to target/common names,
    primary keys become target/common column names.
    """
    return [pk["target_column"] for pk in primary_keys]


def get_common_compare_columns(column_mappings, primary_key_columns):
    """
    Returns non-key columns to compare after source rename.
    These are target/common names.
    """
    all_common_columns = [m["target_column"] for m in column_mappings]
    compare_columns = [col for col in all_common_columns if col not in primary_key_columns]
    return compare_columns


def normalize_compare_columns(df, compare_columns):
    """
    Normalizes compare columns before hashing/comparison.
    Handles:
      - trim spaces
      - cast to string
      - null as empty string
    """
    normalized_df = df
    for col_name in compare_columns:
        normalized_df = normalized_df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull(), F.lit(""))
             .otherwise(F.trim(F.col(col_name).cast("string")))
        )
    return normalized_df


def add_row_hash(df, compare_columns, hash_col_name="row_hash"):
    """
    Adds SHA-256 hash column for non-key compare columns.
    """
    if not compare_columns:
        return df.withColumn(hash_col_name, F.lit(None))

    hash_expr = F.sha2(
        F.concat_ws("||", *[F.coalesce(F.col(c), F.lit("")) for c in compare_columns]),
        256
    )

    return df.withColumn(hash_col_name, hash_expr)


def get_schema_dict(df):
    """
    Returns schema as dict: {column_name: data_type_string}
    """
    return {field.name: field.dataType.simpleString() for field in df.schema.fields}


def spark_df_to_pandas_safe(df, max_rows=50000, label="DataFrame"):
    """
    Safe conversion for SMALL result sets only.
    Prevents driver memory issues in Glue.
    """
    if df is None:
        return pd.DataFrame()

    row_count = df.count()
    print(f"[INFO] {label} row count before toPandas(): {row_count}")

    if row_count > max_rows:
        raise ValueError(
            f"{label} has {row_count} rows which exceeds safe toPandas() limit of {max_rows}. "
            f"Use Spark direct write instead."
        )

    return df.toPandas()


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================
def validate_row_count(source_df, target_df):
    source_count = source_df.count()
    target_count = target_df.count()

    result = {
        "validation_name": "row_count_validation",
        "source_row_count": source_count,
        "target_row_count": target_count,
        "status": "PASS" if source_count == target_count else "FAIL"
    }

    return result


def validate_partition_count(source_df, target_df):
    """
    For now: Spark partition count validation
    (later you can replace with business partition validation)
    """
    source_partitions = source_df.rdd.getNumPartitions()
    target_partitions = target_df.rdd.getNumPartitions()

    result = {
        "validation_name": "partition_count_validation",
        "source_partition_count": source_partitions,
        "target_partition_count": target_partitions,
        "status": "PASS" if source_partitions == target_partitions else "FAIL"
    }

    return result


def validate_duplicate_keys(df, primary_key_columns, dataset_name):
    """
    Checks duplicate keys in a dataset.
    """
    duplicate_keys_df = (
        df.groupBy(*primary_key_columns)
          .count()
          .filter(F.col("count") > 1)
    )

    duplicate_count = duplicate_keys_df.count()

    result = {
        "validation_name": f"{dataset_name.lower()}_duplicate_key_validation",
        "duplicate_key_count": duplicate_count,
        "status": "PASS" if duplicate_count == 0 else "FAIL"
    }

    return result, duplicate_keys_df


def validate_missing_and_extra_by_key(source_df, target_df, primary_key_columns):
    """
    Finds:
      - extra_records_in_source = present in source, absent in target
      - extra_records_in_target = present in target, absent in source
      - matched keys
    """
    source_keys_df = source_df.select(*primary_key_columns).dropDuplicates()
    target_keys_df = target_df.select(*primary_key_columns).dropDuplicates()

    # source-only
    extra_records_in_source_df = source_keys_df.join(
        target_keys_df, on=primary_key_columns, how="left_anti"
    )

    # target-only
    extra_records_in_target_df = target_keys_df.join(
        source_keys_df, on=primary_key_columns, how="left_anti"
    )

    matched_keys_df = source_keys_df.join(
        target_keys_df, on=primary_key_columns, how="inner"
    )

    extra_source_count = extra_records_in_source_df.count()
    extra_target_count = extra_records_in_target_df.count()
    matched_count = matched_keys_df.count()

    result = {
        "validation_name": "missing_extra_by_key_validation",
        "extra_records_in_source_count": extra_source_count,
        "extra_records_in_target_count": extra_target_count,
        "matched_key_count": matched_count,
        "status": "PASS" if extra_source_count == 0 and extra_target_count == 0 else "FAIL"
    }

    return result, extra_records_in_source_df, extra_records_in_target_df, matched_keys_df


def validate_hash_mismatch(source_df, target_df, primary_key_columns, compare_columns):
    """
    For matched keys only:
      - compare row hash of non-key columns
      - if hash differs => mismatch
      - also produce detailed mismatched columns
    """
    # Normalize compare columns before hashing
    source_norm = normalize_compare_columns(source_df, compare_columns)
    target_norm = normalize_compare_columns(target_df, compare_columns)

    # Add row hash
    source_hashed = add_row_hash(source_norm, compare_columns, "source_row_hash")
    target_hashed = add_row_hash(target_norm, compare_columns, "target_row_hash")

    # Select only required columns
    source_select_cols = primary_key_columns + compare_columns + ["source_row_hash"]
    target_select_cols = primary_key_columns + compare_columns + ["target_row_hash"]

    s = source_hashed.select(*source_select_cols).alias("s")
    t = target_hashed.select(*target_select_cols).alias("t")

    # Join on primary keys (only matched records)
    joined_df = s.join(t, on=primary_key_columns, how="inner")

    # Hash mismatches
    hash_mismatch_df = joined_df.filter(F.col("source_row_hash") != F.col("target_row_hash"))
    hash_mismatch_count = hash_mismatch_df.count()

    # Build detailed mismatch columns only for hash-mismatched rows
    mismatch_flag_expressions = []
    for col_name in compare_columns:
        mismatch_flag_expressions.append(
            F.when(
                F.coalesce(F.col(f"s.{col_name}"), F.lit("")) != F.coalesce(F.col(f"t.{col_name}"), F.lit("")),
                F.lit(col_name)
            ).otherwise(F.lit(None))
        )

    detailed_mismatch_df = hash_mismatch_df.withColumn(
        "mismatched_columns",
        F.concat_ws(",", F.array(*mismatch_flag_expressions))
    )

    # Cleaner output
    selected_cols = primary_key_columns.copy()

    for col_name in compare_columns:
        selected_cols.append(F.col(f"s.{col_name}").alias(f"source_{col_name}"))
        selected_cols.append(F.col(f"t.{col_name}").alias(f"target_{col_name}"))

    selected_cols.extend([
        F.col("source_row_hash"),
        F.col("target_row_hash"),
        F.col("mismatched_columns")
    ])

    detailed_mismatch_df = detailed_mismatch_df.select(*selected_cols)

    result = {
        "validation_name": "hash_mismatch_validation",
        "mismatch_count": hash_mismatch_count,
        "status": "PASS" if hash_mismatch_count == 0 else "FAIL"
    }

    return result, detailed_mismatch_df


def validate_schema_and_datatype_drift(source_df, target_df, expected_common_columns):
    """
    Validates schema drift and datatype drift after mapping.

    Checks:
      - column missing in source
      - column missing in target
      - datatype mismatch
      - no drift
    """
    source_schema = get_schema_dict(source_df)
    target_schema = get_schema_dict(target_df)

    drift_rows = []

    for col_name in expected_common_columns:
        source_exists = col_name in source_schema
        target_exists = col_name in target_schema

        source_dtype = source_schema.get(col_name, None)
        target_dtype = target_schema.get(col_name, None)

        if not source_exists:
            drift_type = "MISSING_IN_SOURCE"
            status = "FAIL"
        elif not target_exists:
            drift_type = "MISSING_IN_TARGET"
            status = "FAIL"
        elif source_dtype != target_dtype:
            drift_type = "DATA_TYPE_MISMATCH"
            status = "FAIL"
        else:
            drift_type = "NO_DRIFT"
            status = "PASS"

        drift_rows.append({
            "column_name": col_name,
            "source_exists": source_exists,
            "target_exists": target_exists,
            "source_data_type": source_dtype,
            "target_data_type": target_dtype,
            "drift_type": drift_type,
            "status": status
        })

    drift_df = pd.DataFrame(drift_rows)

    total_columns_checked = len(expected_common_columns)
    total_drift_columns = len(drift_df[drift_df["status"] == "FAIL"])
    datatype_mismatch_count = len(drift_df[drift_df["drift_type"] == "DATA_TYPE_MISMATCH"])
    missing_in_source_count = len(drift_df[drift_df["drift_type"] == "MISSING_IN_SOURCE"])
    missing_in_target_count = len(drift_df[drift_df["drift_type"] == "MISSING_IN_TARGET"])

    result = {
        "validation_name": "schema_datatype_drift_validation",
        "total_columns_checked": total_columns_checked,
        "total_drift_columns": total_drift_columns,
        "datatype_mismatch_count": datatype_mismatch_count,
        "missing_columns_in_source": missing_in_source_count,
        "missing_columns_in_target": missing_in_target_count,
        "status": "PASS" if total_drift_columns == 0 else "FAIL"
    }

    return result, drift_df


# ============================================================
# OUTPUT WRITER FUNCTIONS (VERSION 2 - PRODUCTION SAFE)
# ============================================================
def write_csv_from_pandas(df, output_path):
    """
    Writes a Pandas DataFrame to CSV (local temp path).
    """
    df.to_csv(output_path, index=False)
    print(f"[OK] CSV written locally: {output_path}")


def write_spark_df_as_single_csv(df, output_s3_uri):
    """
    Writes Spark DF directly to S3 as a folder containing a single part file.
    This avoids toPandas() for large datasets.

    Output example:
      s3://bucket/reports/.../detailed_mismatch/
          part-00000-....csv
          _SUCCESS
    """
    delete_s3_prefix(output_s3_uri)

    (
        df.coalesce(1)
          .write
          .mode("overwrite")
          .option("header", "true")
          .csv(output_s3_uri)
    )

    print(f"[OK] Spark CSV written to S3 folder: {output_s3_uri}")


def write_detailed_output_files(
    local_tmp_dir,
    s3_run_output_dir,
    schema_drift_detail_df,
    detailed_mismatch_spark_df,
    extra_records_in_source_spark_df,
    extra_records_in_target_spark_df
):
    """
    VERSION 2:
    - schema drift stays Pandas (small)
    - large outputs use Spark direct write to S3
    """
    # 1) Schema drift CSV (small => safe to keep local+pandas)
    schema_drift_local = os.path.join(local_tmp_dir, "schema_datatype_drift.csv")
    schema_drift_s3 = f"{s3_run_output_dir}schema_datatype_drift.csv"
    write_csv_from_pandas(schema_drift_detail_df, schema_drift_local)
    upload_file_to_s3(schema_drift_local, schema_drift_s3)

    # 2) Detailed mismatch (Spark direct write)
    mismatch_s3_folder = f"{s3_run_output_dir}detailed_mismatch/"
    write_spark_df_as_single_csv(detailed_mismatch_spark_df, mismatch_s3_folder)

    # 3) Extra records in source (Spark direct write)
    extra_source_s3_folder = f"{s3_run_output_dir}extra_records_in_source/"
    write_spark_df_as_single_csv(extra_records_in_source_spark_df, extra_source_s3_folder)

    # 4) Extra records in target (Spark direct write)
    extra_target_s3_folder = f"{s3_run_output_dir}extra_records_in_target/"
    write_spark_df_as_single_csv(extra_records_in_target_spark_df, extra_target_s3_folder)

    return {
        "schema_drift_csv": schema_drift_s3,
        "detailed_mismatch_folder": mismatch_s3_folder,
        "extra_records_in_source_folder": extra_source_s3_folder,
        "extra_records_in_target_folder": extra_target_s3_folder
    }


def generate_excel_report(
    local_tmp_dir,
    s3_run_output_dir,
    source_count,
    target_count,
    total_mismatch,
    extra_records_in_source,
    extra_records_in_target,
    row_count_status,
    partition_count_status,
    schema_drift_status,
    schema_drift_total_columns,
    schema_drift_columns,
    schema_dtype_mismatch_count,
    source_duplicate_keys,
    target_duplicate_keys,
    overall_status,
    schema_drift_detail_df
):
    """
    VERSION 2:
    Keep Excel lightweight and safe.

    IMPORTANT:
    Do NOT put large detailed datasets into Excel in Glue.
    Only keep summary + schema drift detail (small).

    Detailed mismatch / extra records are already written as Spark CSV folders in S3.
    """
    file_name = "reconciliation_report.xlsx"
    local_output_path = os.path.join(local_tmp_dir, file_name)
    s3_output_path = f"{s3_run_output_dir}{file_name}"

    # Summary sheet
    summary_data = [{
        "source_count": source_count,
        "target_count": target_count,
        "total_mismatch": total_mismatch,
        "extra_records_in_source": extra_records_in_source,
        "extra_records_in_target": extra_records_in_target,
        "row_count_status": row_count_status,
        "partition_count_status": partition_count_status,
        "schema_drift_status": schema_drift_status,
        "schema_total_columns_checked": schema_drift_total_columns,
        "schema_drift_columns": schema_drift_columns,
        "schema_datatype_mismatch_count": schema_dtype_mismatch_count,
        "source_duplicate_keys": source_duplicate_keys,
        "target_duplicate_keys": target_duplicate_keys,
        "overall_status": overall_status,
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detailed_mismatch_location": f"{s3_run_output_dir}detailed_mismatch/",
        "extra_records_in_source_location": f"{s3_run_output_dir}extra_records_in_source/",
        "extra_records_in_target_location": f"{s3_run_output_dir}extra_records_in_target/"
    }]
    summary_df = pd.DataFrame(summary_data)

    # Write workbook locally
    with pd.ExcelWriter(local_output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_report", index=False)
        schema_drift_detail_df.to_excel(writer, sheet_name="schema_datatype_drift", index=False)

    print(f"[OK] Excel report generated locally: {local_output_path}")

    # Upload to S3
    upload_file_to_s3(local_output_path, s3_output_path)
    return s3_output_path


def build_summary_report(validation_results):
    """
    Creates a simple Python list summary for console output.
    """
    summary = []
    overall_status = "PASS"

    for result in validation_results:
        summary.append(result)
        if result.get("status") == "FAIL":
            overall_status = "FAIL"

    return overall_status, summary


def print_summary(overall_status, summary):
    print_section("FINAL VALIDATION SUMMARY")
    print(f"Overall Validation Status: {overall_status}")
    print("-" * 110)

    for result in summary:
        print(f"Validation: {result['validation_name']}")
        for k, v in result.items():
            if k != "validation_name":
                print(f"  {k}: {v}")
        print("-" * 110)


# ============================================================
# PLACEHOLDER FOR FUTURE VALIDATIONS
# ============================================================
def run_future_validations():
    """
    Add future validations here later, for example:
      - null threshold validation
      - min/max range checks
      - regex validation
      - referential integrity validation
      - business rule validation
      - aggregate checks
    """
    pass


# ============================================================
# MAIN
# ============================================================
def main():
    spark, glueContext = create_spark_session()

    # Initialize Glue Job
    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)

    try:
        # Create run folder paths
        run_output_dir = create_run_output_dir()
        local_tmp_dir = get_local_tmp_dir()

        print_section("RUN OUTPUT DIRECTORY")
        print(f"S3 Run output directory: {run_output_dir}")
        print(f"Local temp directory: {local_tmp_dir}")

        print_section("READING INPUT FILES")

        # Read inputs
        source_df = read_csv_file(spark, SOURCE_PATH)
        target_df = read_csv_file(spark, TARGET_PATH)
        primary_keys, column_mappings = read_mapping_json(MAPPING_PATH)

        print("Source columns:", source_df.columns)
        print("Target columns:", target_df.columns)

        # ========================================================
        # 1) PRE-MAPPING COLUMN VALIDATION
        # ========================================================
        source_required_columns = [m["source_column"] for m in column_mappings]
        target_required_columns = [m["target_column"] for m in column_mappings]

        print_section("PRE-MAPPING COLUMN VALIDATION")
        source_missing_cols = validate_required_columns(source_df, source_required_columns, "SOURCE")
        target_missing_cols = validate_required_columns(target_df, target_required_columns, "TARGET")

        if source_missing_cols or target_missing_cols:
            raise ValueError("Pre-mapping column validation failed. Please fix input files or mapping JSON.")

        # ========================================================
        # 2) APPLY SOURCE COLUMN MAPPING
        # ========================================================
        print_section("APPLYING SOURCE COLUMN MAPPING")
        source_df = rename_source_columns_to_common_names(source_df, column_mappings)

        print("Source columns after mapping:", source_df.columns)
        print("Target columns:", target_df.columns)

        # ========================================================
        # 3) PREPARE KEYS + COMPARE COLUMNS
        # ========================================================
        primary_key_columns = get_common_primary_keys(primary_keys)
        compare_columns = get_common_compare_columns(column_mappings, primary_key_columns)
        expected_common_columns = [m["target_column"] for m in column_mappings]

        print("Primary key columns:", primary_key_columns)
        print("Compare columns (non-key):", compare_columns)

        # ========================================================
        # 4) POST-MAPPING COMMON COLUMN VALIDATION
        # ========================================================
        print_section("POST-MAPPING COMMON COLUMN VALIDATION")
        source_common_missing = validate_required_columns(source_df, expected_common_columns, "SOURCE (POST-MAPPING)")
        target_common_missing = validate_required_columns(target_df, expected_common_columns, "TARGET")

        if source_common_missing or target_common_missing:
            raise ValueError("Post-mapping common column validation failed.")

        # Optional: show samples
        print_section("SOURCE SAMPLE (POST-MAPPING)")
        source_df.show(5, truncate=False)

        print_section("TARGET SAMPLE")
        target_df.show(5, truncate=False)

        # ========================================================
        # RUN VALIDATIONS
        # ========================================================
        validation_results = []

        # 1) Row count validation
        print_section("RUNNING ROW COUNT VALIDATION")
        row_count_result = validate_row_count(source_df, target_df)
        print(row_count_result)
        validation_results.append(row_count_result)

        # 2) Partition count validation
        print_section("RUNNING PARTITION COUNT VALIDATION")
        partition_count_result = validate_partition_count(source_df, target_df)
        print(partition_count_result)
        validation_results.append(partition_count_result)

        # 3) Schema & Datatype Drift validation
        print_section("RUNNING SCHEMA & DATATYPE DRIFT VALIDATION")
        schema_drift_result, schema_drift_detail_df = validate_schema_and_datatype_drift(
            source_df, target_df, expected_common_columns
        )
        print(schema_drift_result)
        validation_results.append(schema_drift_result)

        print("\nSCHEMA & DATATYPE DRIFT DETAILS:")
        print(schema_drift_detail_df)

        # 4) Duplicate key validation
        print_section("RUNNING DUPLICATE KEY VALIDATION")
        source_dup_result, source_dup_df = validate_duplicate_keys(source_df, primary_key_columns, "SOURCE")
        target_dup_result, target_dup_df = validate_duplicate_keys(target_df, primary_key_columns, "TARGET")

        print(source_dup_result)
        print(target_dup_result)

        validation_results.append(source_dup_result)
        validation_results.append(target_dup_result)

        if source_dup_result["duplicate_key_count"] > 0:
            print("\nSOURCE DUPLICATE KEYS:")
            source_dup_df.show(truncate=False)

        if target_dup_result["duplicate_key_count"] > 0:
            print("\nTARGET DUPLICATE KEYS:")
            target_dup_df.show(truncate=False)

        # 5) Missing / Extra by key
        print_section("RUNNING MISSING / EXTRA BY KEY VALIDATION")
        missing_extra_result, extra_records_in_source_df, extra_records_in_target_df, matched_keys_df = validate_missing_and_extra_by_key(
            source_df, target_df, primary_key_columns
        )

        print(missing_extra_result)
        validation_results.append(missing_extra_result)

        if missing_extra_result["extra_records_in_source_count"] > 0:
            print("\nEXTRA RECORDS IN SOURCE (present in source, absent in target):")
            extra_records_in_source_df.show(truncate=False)

        if missing_extra_result["extra_records_in_target_count"] > 0:
            print("\nEXTRA RECORDS IN TARGET (present in target, absent in source):")
            extra_records_in_target_df.show(truncate=False)

        # 6) Hash mismatch validation
        print_section("RUNNING HASH MISMATCH VALIDATION")
        hash_mismatch_result, detailed_mismatch_df = validate_hash_mismatch(
            source_df, target_df, primary_key_columns, compare_columns
        )

        print(hash_mismatch_result)
        validation_results.append(hash_mismatch_result)

        if hash_mismatch_result["mismatch_count"] > 0:
            print("\nDETAILED MISMATCH RECORDS:")
            detailed_mismatch_df.show(truncate=False)

        # 7) Future validations placeholder
        print_section("FUTURE VALIDATIONS PLACEHOLDER")
        run_future_validations()

        # ========================================================
        # FINAL SUMMARY STATUS
        # ========================================================
        overall_status, summary = build_summary_report(validation_results)

        # ========================================================
        # WRITE DETAILED OUTPUT FILES (CSV)
        # ========================================================
        print_section("WRITING DETAILED OUTPUT FILES (PRODUCTION SAFE)")

        detailed_output_paths = write_detailed_output_files(
            local_tmp_dir=local_tmp_dir,
            s3_run_output_dir=run_output_dir,
            schema_drift_detail_df=schema_drift_detail_df,
            detailed_mismatch_spark_df=detailed_mismatch_df,
            extra_records_in_source_spark_df=extra_records_in_source_df,
            extra_records_in_target_spark_df=extra_records_in_target_df
        )

        print("Detailed output files:")
        for k, v in detailed_output_paths.items():
            print(f"  {k}: {v}")

        # ========================================================
        # GENERATE LIGHTWEIGHT EXCEL REPORT
        # ========================================================
        print_section("GENERATING LIGHTWEIGHT EXCEL REPORT")

        excel_file_path = generate_excel_report(
            local_tmp_dir=local_tmp_dir,
            s3_run_output_dir=run_output_dir,
            source_count=row_count_result["source_row_count"],
            target_count=row_count_result["target_row_count"],
            total_mismatch=hash_mismatch_result["mismatch_count"],
            extra_records_in_source=missing_extra_result["extra_records_in_source_count"],
            extra_records_in_target=missing_extra_result["extra_records_in_target_count"],
            row_count_status=row_count_result["status"],
            partition_count_status=partition_count_result["status"],
            schema_drift_status=schema_drift_result["status"],
            schema_drift_total_columns=schema_drift_result["total_columns_checked"],
            schema_drift_columns=schema_drift_result["total_drift_columns"],
            schema_dtype_mismatch_count=schema_drift_result["datatype_mismatch_count"],
            source_duplicate_keys=source_dup_result["duplicate_key_count"],
            target_duplicate_keys=target_dup_result["duplicate_key_count"],
            overall_status=overall_status,
            schema_drift_detail_df=schema_drift_detail_df
        )

        print(f"Excel report created in S3: {excel_file_path}")

        # ========================================================
        # FINAL SUMMARY (CONSOLE)
        # ========================================================
        print_summary(overall_status, summary)

        # Commit Glue job
        job.commit()

    except Exception as e:
        print(f"[ERROR] Validation job failed: {str(e)}")
        raise

    finally:
        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()