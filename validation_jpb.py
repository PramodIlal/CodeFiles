import json
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# CONFIGURATION (LOCAL FILES)
# ============================================================
SOURCE_PATH = "data/source.csv"
TARGET_PATH = "data/target.csv"
MAPPING_PATH = "data/mapping.json"

# If you want, later you can store outputs in local folder
OUTPUT_BASE_PATH = "output"


# ============================================================
# SPARK SESSION
# ============================================================
def create_spark_session(app_name="ETL Validation Job"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
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
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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
    the common primary key names become target_column names.
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
    This is very useful and should stay.
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
      - missing in target (present in source, absent in target)
      - extra in target (present in target, absent in source)
      - matched keys
    """
    source_keys_df = source_df.select(*primary_key_columns).dropDuplicates()
    target_keys_df = target_df.select(*primary_key_columns).dropDuplicates()

    missing_in_target_df = source_keys_df.join(target_keys_df, on=primary_key_columns, how="left_anti")
    extra_in_target_df = target_keys_df.join(source_keys_df, on=primary_key_columns, how="left_anti")
    matched_keys_df = source_keys_df.join(target_keys_df, on=primary_key_columns, how="inner")

    missing_count = missing_in_target_df.count()
    extra_count = extra_in_target_df.count()
    matched_count = matched_keys_df.count()

    result = {
        "validation_name": "missing_extra_by_key_validation",
        "missing_in_target_count": missing_count,
        "extra_in_target_count": extra_count,
        "matched_key_count": matched_count,
        "status": "PASS" if missing_count == 0 and extra_count == 0 else "FAIL"
    }

    return result, missing_in_target_df, extra_in_target_df, matched_keys_df


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

    # Join on primary keys
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

    # Optional: select cleaner output
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


# ============================================================
# SUMMARY / REPORTING
# ============================================================
def build_summary_report(validation_results):
    """
    Creates a simple Python list summary for now.
    Later you can convert this into a Spark DataFrame or write to CSV/JSON.
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
    print("-" * 80)

    for result in summary:
        print(f"Validation: {result['validation_name']}")
        for k, v in result.items():
            if k != "validation_name":
                print(f"  {k}: {v}")
        print("-" * 80)


# ============================================================
# PLACEHOLDER FOR FUTURE VALIDATIONS (IMPORTANT FOR YOU)
# ============================================================
def run_future_validations():
    """
    Add future validations here later, for example:
      - schema datatype validation
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

    print_section("READING INPUT FILES")

    # Read inputs
    source_df = read_csv_file(spark, SOURCE_PATH)
    target_df = read_csv_file(spark, TARGET_PATH)
    primary_keys, column_mappings = read_mapping_json(MAPPING_PATH)

    print("Source columns:", source_df.columns)
    print("Target columns:", target_df.columns)

    # Step 1: Validate source has required source columns
    source_required_columns = [m["source_column"] for m in column_mappings]
    target_required_columns = [m["target_column"] for m in column_mappings]

    print_section("PRE-MAPPING COLUMN VALIDATION")
    source_missing_cols = validate_required_columns(source_df, source_required_columns, "SOURCE")
    target_missing_cols = validate_required_columns(target_df, target_required_columns, "TARGET")

    if source_missing_cols or target_missing_cols:
        raise ValueError("Pre-mapping column validation failed. Please fix input files or mapping JSON.")

    # Step 2: Rename source columns to common names (target names)
    print_section("APPLYING SOURCE COLUMN MAPPING")
    source_df = rename_source_columns_to_common_names(source_df, column_mappings)

    print("Source columns after mapping:", source_df.columns)
    print("Target columns:", target_df.columns)

    # Step 3: Common keys and compare columns
    primary_key_columns = get_common_primary_keys(primary_keys)
    compare_columns = get_common_compare_columns(column_mappings, primary_key_columns)

    print("Primary key columns:", primary_key_columns)
    print("Compare columns (non-key):", compare_columns)

    # Step 4: Validate common columns exist after mapping
    common_required_columns = [m["target_column"] for m in column_mappings]

    print_section("POST-MAPPING COMMON COLUMN VALIDATION")
    source_common_missing = validate_required_columns(source_df, common_required_columns, "SOURCE (POST-MAPPING)")
    target_common_missing = validate_required_columns(target_df, common_required_columns, "TARGET")

    if source_common_missing or target_common_missing:
        raise ValueError("Post-mapping common column validation failed.")

    # Optional: show small samples
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

    # 3) Duplicate key validation (very useful)
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

    # 4) Missing / Extra by key
    print_section("RUNNING MISSING / EXTRA BY KEY VALIDATION")
    missing_extra_result, missing_in_target_df, extra_in_target_df, matched_keys_df = validate_missing_and_extra_by_key(
        source_df, target_df, primary_key_columns
    )

    print(missing_extra_result)
    validation_results.append(missing_extra_result)

    if missing_extra_result["missing_in_target_count"] > 0:
        print("\nMISSING IN TARGET:")
        missing_in_target_df.show(truncate=False)

    if missing_extra_result["extra_in_target_count"] > 0:
        print("\nEXTRA IN TARGET:")
        extra_in_target_df.show(truncate=False)

    # 5) Hash mismatch validation (for matched keys)
    print_section("RUNNING HASH MISMATCH VALIDATION")
    hash_mismatch_result, detailed_mismatch_df = validate_hash_mismatch(
        source_df, target_df, primary_key_columns, compare_columns
    )

    print(hash_mismatch_result)
    validation_results.append(hash_mismatch_result)

    if hash_mismatch_result["mismatch_count"] > 0:
        print("\nDETAILED MISMATCH RECORDS:")
        detailed_mismatch_df.show(truncate=False)

    # 6) Future validations placeholder
    print_section("FUTURE VALIDATIONS PLACEHOLDER")
    run_future_validations()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    overall_status, summary = build_summary_report(validation_results)
    print_summary(overall_status, summary)

    # Stop Spark
    spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()