import json
import os
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ============================================================
# CONFIGURATION (LOCAL FILES)
# ============================================================
SOURCE_PATH = "data/source.csv"
TARGET_PATH = "data/target.csv"
MAPPING_PATH = "data/mapping.json"
OUTPUT_BASE_PATH = "output"


# ============================================================
# SPARK SESSION
# ============================================================
def create_spark_session(app_name="ETL Validation Job"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        # Disable Arrow to avoid some toPandas() conversion issues on local setups
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ============================================================
# FILE READERS
# ============================================================
def read_csv_file(spark, file_path):
    """
    Reads a CSV file with header and inferSchema enabled.
    """
    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(file_path)
    )


def read_mapping_json(mapping_path):
    """
    Reads mapping JSON from local file system.

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
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

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


def create_run_output_dir(base_output_dir=OUTPUT_BASE_PATH):
    """
    Creates timestamped run folder:
      output/run_YYYYMMDD_HHMMSS/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


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
    Only renames if source column actually exists.
    Example:
      source.first_nm -> first_name
    """
    df = source_df
    existing_cols = set(df.columns)

    for mapping in column_mappings:
        source_col = mapping["source_column"]
        target_col = mapping["target_column"]

        if source_col in existing_cols and source_col != target_col:
            df = df.withColumnRenamed(source_col, target_col)
            existing_cols.remove(source_col)
            existing_cols.add(target_col)

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


def create_empty_key_df(spark, primary_key_columns):
    """
    Creates an empty Spark DF with only primary key columns as string type.
    Useful when validations are skipped.
    """
    schema = T.StructType([
        T.StructField(col_name, T.StringType(), True) for col_name in primary_key_columns
    ])
    return spark.createDataFrame([], schema)


def create_empty_mismatch_df(spark, primary_key_columns, compare_columns):
    """
    Creates an empty Spark DF with columns similar to detailed mismatch output.
    """
    fields = []

    for col_name in primary_key_columns:
        fields.append(T.StructField(col_name, T.StringType(), True))

    for col_name in compare_columns:
        fields.append(T.StructField(f"source_{col_name}", T.StringType(), True))
        fields.append(T.StructField(f"target_{col_name}", T.StringType(), True))

    fields.extend([
        T.StructField("source_row_hash", T.StringType(), True),
        T.StructField("target_row_hash", T.StringType(), True),
        T.StructField("mismatched_columns", T.StringType(), True),
    ])

    schema = T.StructType(fields)
    return spark.createDataFrame([], schema)


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
      - extra_records_in_source = full source rows present in source, absent in target
      - extra_records_in_target = full target rows present in target, absent in source
      - matched keys

    Returns full-row DataFrames for extra_in_source and extra_in_target
    instead of only key columns.
    """
    # Distinct key sets
    source_keys_df = source_df.select(*primary_key_columns).dropDuplicates()
    target_keys_df = target_df.select(*primary_key_columns).dropDuplicates()

    # source-only keys
    extra_source_keys_df = source_keys_df.join(
        target_keys_df, on=primary_key_columns, how="left_anti"
    )

    # target-only keys
    extra_target_keys_df = target_keys_df.join(
        source_keys_df, on=primary_key_columns, how="left_anti"
    )

    # matched keys
    matched_keys_df = source_keys_df.join(
        target_keys_df, on=primary_key_columns, how="inner"
    )

    # FULL source rows for source-only keys
    extra_records_in_source_df = source_df.join(
        extra_source_keys_df, on=primary_key_columns, how="inner"
    )

    # FULL target rows for target-only keys
    extra_records_in_target_df = target_df.join(
        extra_target_keys_df, on=primary_key_columns, how="inner"
    )

    # Counts based on distinct unmatched keys
    extra_source_count = extra_source_keys_df.count()
    extra_target_count = extra_target_keys_df.count()
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
# OUTPUT WRITER FUNCTIONS
# ============================================================
def write_csv_from_pandas(df, output_path):
    """
    Writes a Pandas DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"[OK] CSV written: {output_path}")


def write_spark_df_as_single_csv(df, output_dir_path):
    """
    Writes Spark DF as a single CSV file inside a folder (Spark style).
    Output will be a folder containing:
      - part-00000...
      - _SUCCESS
    """
    (
        df.coalesce(1)
          .write
          .mode("overwrite")
          .option("header", "true")
          .csv(output_dir_path)
    )
    print(f"[OK] Spark CSV folder written: {output_dir_path}")


def write_detailed_output_files(
    run_output_dir,
    schema_drift_detail_df,
    detailed_mismatch_spark_df,
    extra_records_in_source_spark_df,
    extra_records_in_target_spark_df
):
    """
    Writes:
      - schema drift as single CSV file via Pandas (small)
      - detailed outputs as Spark CSV folders (stable for larger data)
    """
    # 1) Schema drift CSV (small, safe with pandas)
    schema_drift_csv_path = os.path.join(run_output_dir, "schema_datatype_drift.csv")
    write_csv_from_pandas(schema_drift_detail_df, schema_drift_csv_path)

    # 2) Detailed mismatch CSV folder
    mismatch_dir = os.path.join(run_output_dir, "detailed_mismatch")
    write_spark_df_as_single_csv(detailed_mismatch_spark_df, mismatch_dir)

    # 3) Extra records in source CSV folder (FULL ROWS)
    extra_source_dir = os.path.join(run_output_dir, "extra_records_in_source")
    write_spark_df_as_single_csv(extra_records_in_source_spark_df, extra_source_dir)

    # 4) Extra records in target CSV folder (FULL ROWS)
    extra_target_dir = os.path.join(run_output_dir, "extra_records_in_target")
    write_spark_df_as_single_csv(extra_records_in_target_spark_df, extra_target_dir)

    return {
        "schema_drift_csv": schema_drift_csv_path,
        "detailed_mismatch_csv_folder": mismatch_dir,
        "extra_records_in_source_csv_folder": extra_source_dir,
        "extra_records_in_target_csv_folder": extra_target_dir
    }


def generate_excel_report(
    run_output_dir,
    source_count,
    target_count,
    total_mismatch,
    extra_records_in_source,
    extra_records_in_target,
    row_count_status,
    partition_count_status,
    pre_mapping_status,
    post_mapping_status,
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
    Generates one Excel file with SMALL sheets only:
      - summary_report
      - schema_datatype_drift

    NOTE:
    Large detailed outputs are written as Spark CSV folders to avoid EOFException.
    """
    file_name = "reconciliation_report.xlsx"
    output_path = os.path.join(run_output_dir, file_name)

    # Summary sheet
    summary_data = [{
        "source_count": source_count,
        "target_count": target_count,
        "total_mismatch": total_mismatch,
        "extra_records_in_source": extra_records_in_source,
        "extra_records_in_target": extra_records_in_target,
        "row_count_status": row_count_status,
        "partition_count_status": partition_count_status,
        "pre_mapping_column_status": pre_mapping_status,
        "post_mapping_column_status": post_mapping_status,
        "schema_drift_status": schema_drift_status,
        "schema_total_columns_checked": schema_drift_total_columns,
        "schema_drift_columns": schema_drift_columns,
        "schema_datatype_mismatch_count": schema_dtype_mismatch_count,
        "source_duplicate_keys": source_duplicate_keys,
        "target_duplicate_keys": target_duplicate_keys,
        "overall_status": overall_status,
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }]
    summary_df = pd.DataFrame(summary_data)

    # Write workbook (small only)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_report", index=False)
        schema_drift_detail_df.to_excel(writer, sheet_name="schema_datatype_drift", index=False)

    print(f"[OK] Excel report generated: {output_path}")
    return output_path


def build_summary_report(validation_results):
    """
    Creates a simple Python list summary for console output.
    SKIPPED is treated as non-pass for overall status because it usually means
    schema issue prevented downstream validation.
    """
    summary = []
    overall_status = "PASS"

    for result in validation_results:
        summary.append(result)
        if result.get("status") != "PASS":
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
    spark = create_spark_session()

    # Create timestamped run folder
    run_output_dir = create_run_output_dir()

    print_section("RUN OUTPUT DIRECTORY")
    print(f"Run output directory: {run_output_dir}")

    print_section("READING INPUT FILES")

    # Read inputs
    source_df_raw = read_csv_file(spark, SOURCE_PATH)
    target_df_raw = read_csv_file(spark, TARGET_PATH)
    primary_keys, column_mappings = read_mapping_json(MAPPING_PATH)

    print("Source columns (raw):", source_df_raw.columns)
    print("Target columns (raw):", target_df_raw.columns)

    # ========================================================
    # 1) PRE-MAPPING COLUMN VALIDATION (NO HARD FAIL)
    # ========================================================
    source_required_columns = [m["source_column"] for m in column_mappings]
    target_required_columns = [m["target_column"] for m in column_mappings]

    print_section("PRE-MAPPING COLUMN VALIDATION")
    source_missing_cols = validate_required_columns(source_df_raw, source_required_columns, "SOURCE")
    target_missing_cols = validate_required_columns(target_df_raw, target_required_columns, "TARGET")

    pre_mapping_columns_ok = not (source_missing_cols or target_missing_cols)
    pre_mapping_result = {
        "validation_name": "pre_mapping_column_validation",
        "source_missing_columns": ",".join(source_missing_cols) if source_missing_cols else "",
        "target_missing_columns": ",".join(target_missing_cols) if target_missing_cols else "",
        "status": "PASS" if pre_mapping_columns_ok else "FAIL"
    }

    if not pre_mapping_columns_ok:
        print("[WARN] Pre-mapping column validation failed. Continuing for schema drift reporting...")

    # ========================================================
    # 2) APPLY SOURCE COLUMN MAPPING (SAFE RENAME)
    # ========================================================
    print_section("APPLYING SOURCE COLUMN MAPPING")
    source_df = rename_source_columns_to_common_names(source_df_raw, column_mappings)
    target_df = target_df_raw

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
    # 4) POST-MAPPING COMMON COLUMN VALIDATION (NO HARD FAIL)
    # ========================================================
    print_section("POST-MAPPING COMMON COLUMN VALIDATION")
    source_common_missing = validate_required_columns(source_df, expected_common_columns, "SOURCE (POST-MAPPING)")
    target_common_missing = validate_required_columns(target_df, expected_common_columns, "TARGET")

    post_mapping_columns_ok = not (source_common_missing or target_common_missing)
    post_mapping_result = {
        "validation_name": "post_mapping_common_column_validation",
        "source_missing_common_columns": ",".join(source_common_missing) if source_common_missing else "",
        "target_missing_common_columns": ",".join(target_common_missing) if target_common_missing else "",
        "status": "PASS" if post_mapping_columns_ok else "FAIL"
    }

    if not post_mapping_columns_ok:
        print("[WARN] Post-mapping common column validation failed. Schema drift will be reported, dependent validations may be skipped.")

    # Optional: show samples
    print_section("SOURCE SAMPLE (POST-MAPPING)")
    source_df.show(5, truncate=False)

    print_section("TARGET SAMPLE")
    target_df.show(5, truncate=False)

    # ========================================================
    # RUN VALIDATIONS
    # ========================================================
    validation_results = []

    # Add pre/post mapping validations
    validation_results.append(pre_mapping_result)
    validation_results.append(post_mapping_result)

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

    # ========================================================
    # DETERMINE WHETHER FULL VALIDATIONS CAN RUN
    # ========================================================
    source_cols_set = set(source_df.columns)
    target_cols_set = set(target_df.columns)
    primary_keys_present = all(
        (pk in source_cols_set) and (pk in target_cols_set)
        for pk in primary_key_columns
    )

    schema_ok_for_full_validation = (
        pre_mapping_columns_ok and
        post_mapping_columns_ok and
        schema_drift_result["status"] == "PASS" and
        primary_keys_present
    )

    # Default placeholders for skipped validations
    source_dup_result = {
        "validation_name": "source_duplicate_key_validation",
        "duplicate_key_count": None,
        "status": "SKIPPED"
    }
    target_dup_result = {
        "validation_name": "target_duplicate_key_validation",
        "duplicate_key_count": None,
        "status": "SKIPPED"
    }
    missing_extra_result = {
        "validation_name": "missing_extra_by_key_validation",
        "extra_records_in_source_count": None,
        "extra_records_in_target_count": None,
        "matched_key_count": None,
        "status": "SKIPPED"
    }
    hash_mismatch_result = {
        "validation_name": "hash_mismatch_validation",
        "mismatch_count": None,
        "status": "SKIPPED"
    }

    source_dup_df = create_empty_key_df(spark, primary_key_columns)
    target_dup_df = create_empty_key_df(spark, primary_key_columns)

    # IMPORTANT: full-row empty placeholders for extra outputs
    extra_records_in_source_df = source_df.limit(0)
    extra_records_in_target_df = target_df.limit(0)

    detailed_mismatch_df = create_empty_mismatch_df(spark, primary_key_columns, compare_columns)

    # ========================================================
    # 4) DUPLICATE KEY + 5) MISSING/EXTRA + 6) HASH MISMATCH
    # ========================================================
    if schema_ok_for_full_validation:
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
            print("\nEXTRA RECORDS IN SOURCE (FULL ROWS present in source, absent in target):")
            extra_records_in_source_df.show(truncate=False)

        if missing_extra_result["extra_records_in_target_count"] > 0:
            print("\nEXTRA RECORDS IN TARGET (FULL ROWS present in target, absent in source):")
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

    else:
        print_section("SKIPPING DEPENDENT VALIDATIONS")
        print("Schema/pre-mapping/post-mapping validation failed OR primary key missing.")
        print("So duplicate key / missing-extra / hash mismatch validations are skipped safely.")

        validation_results.append(source_dup_result)
        validation_results.append(target_dup_result)
        validation_results.append(missing_extra_result)
        validation_results.append(hash_mismatch_result)

    # 7) Future validations placeholder
    print_section("FUTURE VALIDATIONS PLACEHOLDER")
    run_future_validations()

    # ========================================================
    # FINAL SUMMARY STATUS
    # ========================================================
    overall_status, summary = build_summary_report(validation_results)

    # ========================================================
    # SAFE SUMMARY VALUES FOR REPORTING
    # ========================================================
    total_mismatch_for_report = (
        hash_mismatch_result["mismatch_count"]
        if hash_mismatch_result["mismatch_count"] is not None else 0
    )

    extra_source_for_report = (
        missing_extra_result["extra_records_in_source_count"]
        if missing_extra_result["extra_records_in_source_count"] is not None else 0
    )

    extra_target_for_report = (
        missing_extra_result["extra_records_in_target_count"]
        if missing_extra_result["extra_records_in_target_count"] is not None else 0
    )

    source_dup_for_report = (
        source_dup_result["duplicate_key_count"]
        if source_dup_result["duplicate_key_count"] is not None else 0
    )

    target_dup_for_report = (
        target_dup_result["duplicate_key_count"]
        if target_dup_result["duplicate_key_count"] is not None else 0
    )

    # ========================================================
    # WRITE DETAILED OUTPUT FILES
    # ========================================================
    print_section("WRITING DETAILED OUTPUT FILES")

    detailed_output_paths = write_detailed_output_files(
        run_output_dir=run_output_dir,
        schema_drift_detail_df=schema_drift_detail_df,
        detailed_mismatch_spark_df=detailed_mismatch_df,
        extra_records_in_source_spark_df=extra_records_in_source_df,
        extra_records_in_target_spark_df=extra_records_in_target_df
    )

    print("Detailed output files:")
    for k, v in detailed_output_paths.items():
        print(f"  {k}: {v}")

    # ========================================================
    # GENERATE EXCEL REPORT (SMALL SHEETS ONLY)
    # ========================================================
    print_section("GENERATING EXCEL REPORT")

    excel_file_path = generate_excel_report(
        run_output_dir=run_output_dir,
        source_count=row_count_result["source_row_count"],
        target_count=row_count_result["target_row_count"],
        total_mismatch=total_mismatch_for_report,
        extra_records_in_source=extra_source_for_report,
        extra_records_in_target=extra_target_for_report,
        row_count_status=row_count_result["status"],
        partition_count_status=partition_count_result["status"],
        pre_mapping_status=pre_mapping_result["status"],
        post_mapping_status=post_mapping_result["status"],
        schema_drift_status=schema_drift_result["status"],
        schema_drift_total_columns=schema_drift_result["total_columns_checked"],
        schema_drift_columns=schema_drift_result["total_drift_columns"],
        schema_dtype_mismatch_count=schema_drift_result["datatype_mismatch_count"],
        source_duplicate_keys=source_dup_for_report,
        target_duplicate_keys=target_dup_for_report,
        overall_status=overall_status,
        schema_drift_detail_df=schema_drift_detail_df
    )

    print(f"Excel report created: {excel_file_path}")

    # ========================================================
    # FINAL SUMMARY (CONSOLE)
    # ========================================================
    print_summary(overall_status, summary)

    # Stop Spark
    spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
