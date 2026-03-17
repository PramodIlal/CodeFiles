import json
import os
from datetime import datetime
from urllib.parse import urlparse

import boto3
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType

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

SOURCE_PATH = args["SOURCE_PATH"]       # DynamoDB table name
TARGET_PATH = args["TARGET_PATH"]       # DynamoDB table name
MAPPING_PATH = args["MAPPING_PATH"]     # S3 path to mapping JSON
REPORT_BASE_PATH = args["REPORT_BASE_PATH"]
RUN_ID = args["RUN_ID"]


# ============================================================
# INTERNAL TECHNICAL COLUMN NAMES
# ============================================================
SOURCE_ROW_ID_COL = "__src_row_id"
TARGET_ROW_ID_COL = "__tgt_row_id"


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


def upload_json_to_s3(data, s3_uri):
    s3 = boto3.client("s3")
    bucket, key = parse_s3_uri(s3_uri)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json"
    )
    print(f"[OK] JSON uploaded to S3: {s3_uri}")


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
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    keys = list_s3_objects(bucket, prefix)

    if not keys:
        return

    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
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
def read_dynamodb_table(glueContext, table_name):
    """
    Reads full DynamoDB table into Spark DataFrame.
    """
    dyf = glueContext.create_dynamic_frame.from_options(
        connection_type="dynamodb",
        connection_options={
            "dynamodb.input.tableName": table_name,
            "dynamodb.throughput.read.percent": "0.5"
        }
    )
    return dyf.toDF()


def read_mapping_json(mapping_path):
    """
    Recommended format:
    {
      "primary_keys": [
        {"source_column": "id", "target_column": "cust_id", "data_type": "numeric"}
      ],
      "column_mappings": [
        {"source_column": "id", "target_column": "cust_id", "data_type": "numeric"},
        {"source_column": "price", "target_column": "price_amt", "data_type": "numeric"},
        {"source_column": "first_nm", "target_column": "first_name", "data_type": "string"}
      ]
    }

    Notes:
    - data_type optional; defaults to "string"
    """
    content = read_text_from_s3(mapping_path)
    mapping = json.loads(content)

    primary_keys = mapping.get("primary_keys", [])
    column_mappings = mapping.get("column_mappings", [])

    if not primary_keys:
        raise ValueError("Mapping JSON must contain 'primary_keys'.")

    if not column_mappings:
        raise ValueError("Mapping JSON must contain 'column_mappings'.")

    for pk in primary_keys:
        pk["data_type"] = pk.get("data_type", "string").lower()

    for col in column_mappings:
        col["data_type"] = col.get("data_type", "string").lower()

    return primary_keys, column_mappings


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def print_section(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def create_run_output_dir(base_output_dir=REPORT_BASE_PATH, run_id=RUN_ID):
    if not base_output_dir.endswith("/"):
        base_output_dir += "/"
    return f"{base_output_dir}run_id={run_id}/"


def get_local_tmp_dir(run_id=RUN_ID):
    local_tmp_dir = f"/tmp/validation_run_{run_id}"
    os.makedirs(local_tmp_dir, exist_ok=True)
    return local_tmp_dir


def validate_required_columns(df, required_columns, dataset_name):
    df_columns = set(df.columns)
    missing_columns = [col for col in required_columns if col not in df_columns]

    if missing_columns:
        print(f"[ERROR] Missing columns in {dataset_name}: {missing_columns}")
    else:
        print(f"[OK] All required columns are present in {dataset_name}")

    return missing_columns


def rename_target_columns_to_common_names(target_df, column_mappings):
    """
    Rename target columns to source/common names.
    """
    df = target_df
    for mapping in column_mappings:
        source_col = mapping["source_column"]
        target_col = mapping["target_column"]

        if target_col != source_col:
            df = df.withColumnRenamed(target_col, source_col)

    return df


def get_common_primary_keys(primary_keys):
    return [pk["source_column"] for pk in primary_keys]


def get_common_compare_columns(column_mappings, primary_key_columns):
    all_common_columns = [m["source_column"] for m in column_mappings]
    return [col for col in all_common_columns if col not in primary_key_columns]


def get_column_mapping_by_source(column_mappings):
    return {m["source_column"]: m for m in column_mappings}


def get_primary_key_mapping_by_source(primary_keys):
    return {pk["source_column"]: pk for pk in primary_keys}


def get_data_type_for_column(column_name, mapping_by_source, default_type="string"):
    mapping = mapping_by_source.get(column_name, {})
    return mapping.get("data_type", default_type).lower()


def get_schema_dict(df):
    """
    Returns schema as dict: {column_name: data_type_string}
    """
    return {field.name: field.dataType.simpleString() for field in df.schema.fields}


def get_empty_df_with_schema(spark, schema):
    return spark.createDataFrame([], schema)


def get_normalized_pk_col_name(pk_col):
    return f"__norm_pk_{pk_col}"


def add_row_id(df, row_id_col):
    """
    Adds stable technical row id before normalization.
    This is used ONLY for safe raw-value retrieval later.
    """
    if row_id_col in df.columns:
        return df
    return df.withColumn(row_id_col, F.monotonically_increasing_id())


# ============================================================
# DATATYPE / NORMALIZATION HELPERS
# ============================================================
def get_spark_type_family(spark_type_str):
    if spark_type_str is None:
        return "unknown"

    t = spark_type_str.lower()

    if t in {"byte", "short", "int", "integer", "bigint", "long"}:
        return "integer"

    if t in {"float", "double"}:
        return "floating"

    if t.startswith("decimal"):
        return "decimal"

    if t in {"string", "varchar", "char"}:
        return "string"

    if t == "boolean":
        return "boolean"

    if t == "date":
        return "date"

    if t == "timestamp":
        return "timestamp"

    return "other"


def is_spark_type_compatible_with_declared_type(spark_type_str, declared_type):
    family = get_spark_type_family(spark_type_str)
    declared_type = (declared_type or "string").lower()

    if declared_type == "numeric":
        return family in {"integer", "floating", "decimal"}

    if declared_type == "string":
        return family == "string"

    if declared_type == "boolean":
        return family == "boolean"

    if declared_type == "date":
        return family in {"date", "timestamp", "string"}  # relaxed

    if declared_type == "timestamp":
        return family in {"timestamp", "string"}  # relaxed

    return True


def get_normalized_expression(col_name, declared_type):
    """
    Returns a Spark Column expression for normalized comparison.
    """
    declared_type = (declared_type or "string").lower()

    if declared_type == "numeric":
        return (
            F.when(F.col(col_name).isNull(), F.lit(""))
             .otherwise(
                F.col(col_name)
                 .cast("decimal(38,10)")
                 .cast("string")
             )
        )

    return (
        F.when(F.col(col_name).isNull(), F.lit(""))
         .otherwise(F.trim(F.col(col_name).cast("string")))
    )


def normalize_column_for_comparison(df, col_name, declared_type):
    """
    Overwrites same column with normalized value.
    Used for compare columns only in working DF.
    """
    return df.withColumn(col_name, get_normalized_expression(col_name, declared_type))


def add_normalized_pk_columns(df, primary_key_columns, pk_mapping_by_source):
    """
    Adds internal normalized PK columns WITHOUT changing original PK values.
    Example:
      id -> __norm_pk_id
    """
    work_df = df
    for pk in primary_key_columns:
        declared_type = get_data_type_for_column(pk, pk_mapping_by_source)
        norm_pk_col = get_normalized_pk_col_name(pk)

        work_df = work_df.withColumn(
            norm_pk_col,
            get_normalized_expression(pk, declared_type)
        )

    return work_df


def normalize_compare_columns(df, compare_columns, column_mappings_by_source):
    """
    Normalizes compare columns IN PLACE in working DF.
    Raw DF remains untouched separately.
    """
    work_df = df
    for col_name in compare_columns:
        declared_type = get_data_type_for_column(col_name, column_mappings_by_source)
        work_df = normalize_column_for_comparison(work_df, col_name, declared_type)

    return work_df


def build_working_df(df, primary_key_columns, compare_columns, pk_mapping_by_source, column_mappings_by_source):
    """
    Build working dataframe:
      - original PK columns kept as-is
      - stable row id retained
      - add internal normalized PK columns (__norm_pk_*)
      - normalize compare columns in-place
    """
    work_df = add_normalized_pk_columns(df, primary_key_columns, pk_mapping_by_source)
    work_df = normalize_compare_columns(work_df, compare_columns, column_mappings_by_source)
    return work_df


def add_row_hash(df, compare_columns, hash_col_name="row_hash"):
    """
    Adds SHA-256 hash column for already-normalized compare columns.
    """
    if not compare_columns:
        return df.withColumn(hash_col_name, F.lit(None).cast("string"))

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

    return {
        "validation_name": "row_count_validation",
        "source_row_count": source_count,
        "target_row_count": target_count,
        "status": "PASS" if source_count == target_count else "FAIL"
    }


def validate_partition_count(source_df, target_df):
    source_partitions = source_df.rdd.getNumPartitions()
    target_partitions = target_df.rdd.getNumPartitions()

    return {
        "validation_name": "partition_count_validation",
        "source_partition_count": source_partitions,
        "target_partition_count": target_partitions,
        "status": "PASS" if source_partitions == target_partitions else "FAIL"
    }


def validate_duplicate_keys(work_df, primary_key_columns, dataset_name):
    """
    Uses normalized PK columns internally for duplicate detection,
    but outputs ORIGINAL PK columns in report.
    """
    norm_pk_cols = [get_normalized_pk_col_name(pk) for pk in primary_key_columns]

    duplicate_norm_keys_df = (
        work_df.groupBy(*norm_pk_cols)
               .count()
               .filter(F.col("count") > 1)
    )

    duplicate_count = duplicate_norm_keys_df.count()

    if duplicate_count > 0:
        duplicate_keys_df = (
            work_df.alias("w")
            .join(
                duplicate_norm_keys_df.alias("d"),
                on=norm_pk_cols,
                how="inner"
            )
            .select(
                *[F.col(f"w.{pk}").cast("string").alias(pk) for pk in primary_key_columns]
            )
            .dropDuplicates()
        )
    else:
        schema = StructType([StructField(pk, StringType(), True) for pk in primary_key_columns])
        duplicate_keys_df = get_empty_df_with_schema(work_df.sparkSession, schema)

    result = {
        "validation_name": f"{dataset_name.lower()}_duplicate_key_validation",
        "duplicate_key_count": duplicate_count,
        "status": "PASS" if duplicate_count == 0 else "FAIL"
    }

    return result, duplicate_keys_df


def validate_missing_and_extra_by_key(source_work_df, target_work_df, primary_key_columns):
    """
    Uses normalized PK columns internally for matching,
    but outputs ORIGINAL PK columns in reports.
    """
    norm_pk_cols = [get_normalized_pk_col_name(pk) for pk in primary_key_columns]

    source_keys_df = (
        source_work_df
        .select(
            *[F.col(pk).cast("string").alias(pk) for pk in primary_key_columns],
            *[F.col(norm_pk).alias(norm_pk) for norm_pk in norm_pk_cols]
        )
        .dropDuplicates(norm_pk_cols)
    )

    target_keys_df = (
        target_work_df
        .select(
            *[F.col(pk).cast("string").alias(pk) for pk in primary_key_columns],
            *[F.col(norm_pk).alias(norm_pk) for norm_pk in norm_pk_cols]
        )
        .dropDuplicates(norm_pk_cols)
    )

    extra_records_in_source_df = (
        source_keys_df.alias("s")
        .join(target_keys_df.alias("t"), on=norm_pk_cols, how="left_anti")
        .select(*[F.col(f"s.{pk}").alias(pk) for pk in primary_key_columns])
    )

    extra_records_in_target_df = (
        target_keys_df.alias("t")
        .join(source_keys_df.alias("s"), on=norm_pk_cols, how="left_anti")
        .select(*[F.col(f"t.{pk}").alias(pk) for pk in primary_key_columns])
    )

    matched_keys_df = (
        source_keys_df.alias("s")
        .join(target_keys_df.alias("t"), on=norm_pk_cols, how="inner")
        .select(
            *[F.col(f"s.{pk}").alias(pk) for pk in primary_key_columns],
            *[F.col(f"s.{norm_pk}").alias(norm_pk) for norm_pk in norm_pk_cols]
        )
        .dropDuplicates(norm_pk_cols)
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


def validate_hash_mismatch(source_df, target_df, source_work_df, target_work_df, primary_key_columns, compare_columns):
    """
    Version 3.5 (Enterprise safest):
    - Uses normalized PK columns for joins
    - Uses normalized compare columns for hash/value comparison
    - Uses stable technical row ids for raw value lookup
    - Reports ORIGINAL/raw source and target values in detailed mismatch report

    Detailed mismatch report format:
      [primary keys..., column_name, source_value, target_value]
    """
    mismatch_schema = StructType(
        [StructField(pk, StringType(), True) for pk in primary_key_columns] +
        [
            StructField("column_name", StringType(), True),
            StructField("source_value", StringType(), True),
            StructField("target_value", StringType(), True),
        ]
    )

    if not compare_columns:
        result = {
            "validation_name": "hash_mismatch_validation",
            "mismatch_count": 0,
            "mismatch_record_count": 0,
            "mismatch_field_count": 0,
            "status": "PASS"
        }
        empty_df = get_empty_df_with_schema(source_df.sparkSession, mismatch_schema)
        return result, empty_df

    norm_pk_cols = [get_normalized_pk_col_name(pk) for pk in primary_key_columns]

    # Add row hash on normalized compare columns
    source_hashed = add_row_hash(source_work_df, compare_columns, "source_row_hash")
    target_hashed = add_row_hash(target_work_df, compare_columns, "target_row_hash")

    # Select normalized working columns for comparison + row ids
    source_work_select_cols = [SOURCE_ROW_ID_COL] + primary_key_columns + norm_pk_cols + compare_columns + ["source_row_hash"]
    target_work_select_cols = [TARGET_ROW_ID_COL] + primary_key_columns + norm_pk_cols + compare_columns + ["target_row_hash"]

    sw = source_hashed.select(*source_work_select_cols).alias("sw")
    tw = target_hashed.select(*target_work_select_cols).alias("tw")

    # Join on normalized PKs
    joined_work_df = sw.join(tw, on=norm_pk_cols, how="inner")

    # Row-level mismatch detection
    hash_mismatch_df = joined_work_df.filter(F.col("source_row_hash") != F.col("target_row_hash"))
    mismatch_record_count = hash_mismatch_df.count()

    if mismatch_record_count == 0:
        result = {
            "validation_name": "hash_mismatch_validation",
            "mismatch_count": 0,
            "mismatch_record_count": 0,
            "mismatch_field_count": 0,
            "status": "PASS"
        }
        empty_df = get_empty_df_with_schema(source_df.sparkSession, mismatch_schema)
        return result, empty_df

    # Raw source/target aliases with stable row ids
    s_raw = source_df.alias("s")
    t_raw = target_df.alias("t")

    # SAFEST PART:
    # Rejoin raw rows using technical row ids, NOT raw PKs
    mismatch_with_raw_df = (
        hash_mismatch_df.alias("m")
        .join(
            s_raw,
            on=F.col(f"m.sw.{SOURCE_ROW_ID_COL}") == F.col(f"s.{SOURCE_ROW_ID_COL}"),
            how="inner"
        )
        .join(
            t_raw,
            on=F.col(f"m.tw.{TARGET_ROW_ID_COL}") == F.col(f"t.{TARGET_ROW_ID_COL}"),
            how="inner"
        )
    )

    # Field-level mismatch rows using normalized compare columns for detection
    # but raw source/target values for output
    mismatch_dfs = []

    for col_name in compare_columns:
        per_col_mismatch_df = (
            mismatch_with_raw_df
            .filter(
                F.coalesce(F.col(f"m.sw.{col_name}"), F.lit("")) !=
                F.coalesce(F.col(f"m.tw.{col_name}"), F.lit(""))
            )
            .select(
                *[F.col(f"m.sw.{pk}").cast("string").alias(pk) for pk in primary_key_columns],
                F.lit(col_name).alias("column_name"),
                F.col(f"s.{col_name}").cast("string").alias("source_value"),
                F.col(f"t.{col_name}").cast("string").alias("target_value")
            )
        )
        mismatch_dfs.append(per_col_mismatch_df)

    if mismatch_dfs:
        detailed_mismatch_df = mismatch_dfs[0]
        for df_part in mismatch_dfs[1:]:
            detailed_mismatch_df = detailed_mismatch_df.unionByName(df_part)
    else:
        detailed_mismatch_df = get_empty_df_with_schema(source_df.sparkSession, mismatch_schema)

    mismatch_field_count = detailed_mismatch_df.count()

    result = {
        "validation_name": "hash_mismatch_validation",
        "mismatch_count": mismatch_record_count,  # backward compatibility
        "mismatch_record_count": mismatch_record_count,
        "mismatch_field_count": mismatch_field_count,
        "status": "PASS" if mismatch_record_count == 0 else "FAIL"
    }

    return result, detailed_mismatch_df


def validate_schema_and_datatype_drift(source_df, target_df, column_mappings):
    """
    Mapping-driven schema + datatype drift validation.

    Status rules:
      PASS:
        - both source and target actual types are compatible with declared type
        - and exact Spark types match
      WARN:
        - both are compatible with declared type
        - but exact Spark types differ
      FAIL:
        - missing column
        - or either side incompatible with declared type
    """
    source_schema = get_schema_dict(source_df)
    target_schema = get_schema_dict(target_df)

    drift_rows = []

    for mapping in column_mappings:
        col_name = mapping["source_column"]
        declared_type = mapping.get("data_type", "string").lower()

        source_exists = col_name in source_schema
        target_exists = col_name in target_schema

        source_dtype = source_schema.get(col_name)
        target_dtype = target_schema.get(col_name)

        if not source_exists:
            drift_type = "MISSING_IN_SOURCE"
            status = "FAIL"
        elif not target_exists:
            drift_type = "MISSING_IN_TARGET"
            status = "FAIL"
        else:
            source_compatible = is_spark_type_compatible_with_declared_type(source_dtype, declared_type)
            target_compatible = is_spark_type_compatible_with_declared_type(target_dtype, declared_type)

            if not source_compatible or not target_compatible:
                drift_type = "DECLARED_TYPE_INCOMPATIBLE"
                status = "FAIL"
            elif source_dtype == target_dtype:
                drift_type = "NO_DRIFT"
                status = "PASS"
            else:
                drift_type = "COMPATIBLE_TYPE_VARIATION"
                status = "WARN"

        drift_rows.append({
            "column_name": col_name,
            "declared_data_type": declared_type,
            "source_exists": source_exists,
            "target_exists": target_exists,
            "source_data_type": source_dtype,
            "target_data_type": target_dtype,
            "drift_type": drift_type,
            "status": status
        })

    drift_df = pd.DataFrame(drift_rows)

    total_columns_checked = len(column_mappings)
    total_fail_columns = len(drift_df[drift_df["status"] == "FAIL"])
    total_warn_columns = len(drift_df[drift_df["status"] == "WARN"])
    compatible_type_variation_count = len(drift_df[drift_df["drift_type"] == "COMPATIBLE_TYPE_VARIATION"])
    declared_type_incompatible_count = len(drift_df[drift_df["drift_type"] == "DECLARED_TYPE_INCOMPATIBLE"])
    missing_in_source_count = len(drift_df[drift_df["drift_type"] == "MISSING_IN_SOURCE"])
    missing_in_target_count = len(drift_df[drift_df["drift_type"] == "MISSING_IN_TARGET"])

    result = {
        "validation_name": "schema_datatype_drift_validation",
        "total_columns_checked": total_columns_checked,
        "total_fail_columns": total_fail_columns,
        "total_warn_columns": total_warn_columns,
        "compatible_type_variation_count": compatible_type_variation_count,
        "declared_type_incompatible_count": declared_type_incompatible_count,
        "missing_columns_in_source": missing_in_source_count,
        "missing_columns_in_target": missing_in_target_count,
        "status": "PASS" if total_fail_columns == 0 else "FAIL"
    }

    return result, drift_df


# ============================================================
# OUTPUT WRITER FUNCTIONS (PRODUCTION SAFE)
# ============================================================
def write_csv_from_pandas(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"[OK] CSV written locally: {output_path}")


def write_empty_csv_to_s3(s3_uri, columns):
    bucket, key = parse_s3_uri(s3_uri)

    if key.endswith("/"):
        key = key + "empty.csv"

    csv_header = ",".join(columns) + "\n"

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_header.encode("utf-8"),
        ContentType="text/csv"
    )

    print(f"[OK] Header-only empty CSV written to S3: s3://{bucket}/{key}")


def write_spark_df_as_single_csv(df, output_s3_uri):
    delete_s3_prefix(output_s3_uri)

    row_count = df.count()
    print(f"[INFO] Writing Spark DF to S3. Row count = {row_count}, Path = {output_s3_uri}")

    if row_count == 0:
        write_empty_csv_to_s3(output_s3_uri, df.columns)
        return

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
    extra_records_in_target_spark_df,
    source_dup_df,
    target_dup_df
):
    schema_drift_local = os.path.join(local_tmp_dir, "schema_datatype_drift.csv")
    schema_drift_s3 = f"{s3_run_output_dir}schema_datatype_drift.csv"
    write_csv_from_pandas(schema_drift_detail_df, schema_drift_local)
    upload_file_to_s3(schema_drift_local, schema_drift_s3)

    mismatch_s3_folder = f"{s3_run_output_dir}detailed_mismatch/"
    write_spark_df_as_single_csv(detailed_mismatch_spark_df, mismatch_s3_folder)

    extra_source_s3_folder = f"{s3_run_output_dir}extra_records_in_source/"
    write_spark_df_as_single_csv(extra_records_in_source_spark_df, extra_source_s3_folder)

    extra_target_s3_folder = f"{s3_run_output_dir}extra_records_in_target/"
    write_spark_df_as_single_csv(extra_records_in_target_spark_df, extra_target_s3_folder)

    source_dup_s3_folder = f"{s3_run_output_dir}source_duplicate_keys/"
    write_spark_df_as_single_csv(source_dup_df, source_dup_s3_folder)

    target_dup_s3_folder = f"{s3_run_output_dir}target_duplicate_keys/"
    write_spark_df_as_single_csv(target_dup_df, target_dup_s3_folder)

    return {
        "schema_drift_csv": schema_drift_s3,
        "detailed_mismatch_folder": mismatch_s3_folder,
        "extra_records_in_source_folder": extra_source_s3_folder,
        "extra_records_in_target_folder": extra_target_s3_folder,
        "source_duplicate_keys_folder": source_dup_s3_folder,
        "target_duplicate_keys_folder": target_dup_s3_folder
    }


def generate_excel_report(
    local_tmp_dir,
    s3_run_output_dir,
    source_count,
    target_count,
    mismatch_record_count,
    mismatch_field_count,
    extra_records_in_source,
    extra_records_in_target,
    row_count_status,
    partition_count_status,
    schema_drift_status,
    schema_drift_total_columns,
    schema_fail_columns,
    schema_warn_columns,
    source_duplicate_keys,
    target_duplicate_keys,
    overall_status,
    schema_drift_detail_df
):
    file_name = "reconciliation_report.xlsx"
    local_output_path = os.path.join(local_tmp_dir, file_name)
    s3_output_path = f"{s3_run_output_dir}{file_name}"

    summary_data = [{
        "source_count": source_count,
        "target_count": target_count,
        "mismatch_record_count": mismatch_record_count,
        "mismatch_field_count": mismatch_field_count,
        "extra_records_in_source": extra_records_in_source,
        "extra_records_in_target": extra_records_in_target,
        "row_count_status": row_count_status,
        "partition_count_status": partition_count_status,
        "schema_drift_status": schema_drift_status,
        "schema_total_columns_checked": schema_drift_total_columns,
        "schema_fail_columns": schema_fail_columns,
        "schema_warn_columns": schema_warn_columns,
        "source_duplicate_keys": source_duplicate_keys,
        "target_duplicate_keys": target_duplicate_keys,
        "overall_status": overall_status,
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detailed_mismatch_location": f"{s3_run_output_dir}detailed_mismatch/",
        "extra_records_in_source_location": f"{s3_run_output_dir}extra_records_in_source/",
        "extra_records_in_target_location": f"{s3_run_output_dir}extra_records_in_target/",
        "source_duplicate_keys_location": f"{s3_run_output_dir}source_duplicate_keys/",
        "target_duplicate_keys_location": f"{s3_run_output_dir}target_duplicate_keys/"
    }]
    summary_df = pd.DataFrame(summary_data)

    with pd.ExcelWriter(local_output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_report", index=False)
        schema_drift_detail_df.to_excel(writer, sheet_name="schema_datatype_drift", index=False)

    print(f"[OK] Excel report generated locally: {local_output_path}")

    upload_file_to_s3(local_output_path, s3_output_path)
    return s3_output_path


def write_manifest_json(
    s3_run_output_dir,
    run_id,
    source_table,
    target_table,
    mapping_path,
    overall_status,
    validation_summary,
    detailed_output_paths,
    excel_report_path
):
    manifest = {
        "run_id": run_id,
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_table": source_table,
        "target_table": target_table,
        "mapping_path": mapping_path,
        "overall_status": overall_status,
        "validation_summary": validation_summary,
        "output_files": {
            **detailed_output_paths,
            "excel_report": excel_report_path
        }
    }

    manifest_s3_path = f"{s3_run_output_dir}manifest.json"
    upload_json_to_s3(manifest, manifest_s3_path)
    return manifest_s3_path


def build_summary_report(validation_results):
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
        run_output_dir = create_run_output_dir()
        local_tmp_dir = get_local_tmp_dir()

        print_section("RUN OUTPUT DIRECTORY")
        print(f"S3 Run output directory: {run_output_dir}")
        print(f"Local temp directory: {local_tmp_dir}")

        print_section("READING INPUT FILES")

        # Read inputs
        source_df = read_dynamodb_table(glueContext, SOURCE_PATH)
        target_df = read_dynamodb_table(glueContext, TARGET_PATH)
        primary_keys, column_mappings = read_mapping_json(MAPPING_PATH)

        # Add stable technical row ids BEFORE any normalization
        source_df = add_row_id(source_df, SOURCE_ROW_ID_COL)
        target_df = add_row_id(target_df, TARGET_ROW_ID_COL)

        # Mapping lookups
        column_mappings_by_source = get_column_mapping_by_source(column_mappings)
        pk_mapping_by_source = get_primary_key_mapping_by_source(primary_keys)

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
        # 2) APPLY TARGET COLUMN MAPPING
        # ========================================================
        print_section("APPLYING TARGET COLUMN MAPPING")
        target_df = rename_target_columns_to_common_names(target_df, column_mappings)

        print("Source columns:", source_df.columns)
        print("Target columns after mapping:", target_df.columns)

        # ========================================================
        # 3) PREPARE KEYS + COMPARE COLUMNS
        # ========================================================
        primary_key_columns = get_common_primary_keys(primary_keys)
        compare_columns = get_common_compare_columns(column_mappings, primary_key_columns)
        expected_common_columns = [m["source_column"] for m in column_mappings]

        print("Primary key columns:", primary_key_columns)
        print("Compare columns (non-key):", compare_columns)

        # ========================================================
        # 4) POST-MAPPING COMMON COLUMN VALIDATION
        # ========================================================
        print_section("POST-MAPPING COMMON COLUMN VALIDATION")
        source_common_missing = validate_required_columns(source_df, expected_common_columns, "SOURCE")
        target_common_missing = validate_required_columns(target_df, expected_common_columns, "TARGET (POST-MAPPING)")

        if source_common_missing or target_common_missing:
            raise ValueError("Post-mapping common column validation failed.")

        # Optional samples
        print_section("SOURCE SAMPLE")
        source_df.show(5, truncate=False)

        print_section("TARGET SAMPLE (POST-MAPPING)")
        target_df.show(5, truncate=False)

        # ========================================================
        # 5) BUILD NORMALIZED WORKING DATAFRAMES (VERSION 3.5)
        # ========================================================
        print_section("BUILDING NORMALIZED WORKING DATAFRAMES")

        source_work_df = build_working_df(
            df=source_df,
            primary_key_columns=primary_key_columns,
            compare_columns=compare_columns,
            pk_mapping_by_source=pk_mapping_by_source,
            column_mappings_by_source=column_mappings_by_source
        )

        target_work_df = build_working_df(
            df=target_df,
            primary_key_columns=primary_key_columns,
            compare_columns=compare_columns,
            pk_mapping_by_source=pk_mapping_by_source,
            column_mappings_by_source=column_mappings_by_source
        )

        print("[OK] Working DataFrames created with stable row ids + normalized internal PKs + normalized compare columns")

        # ========================================================
        # RUN VALIDATIONS
        # ========================================================
        validation_results = []

        # 1) Row count validation (raw df okay)
        print_section("RUNNING ROW COUNT VALIDATION")
        row_count_result = validate_row_count(source_df, target_df)
        print(row_count_result)
        validation_results.append(row_count_result)

        # 2) Partition count validation (raw df okay)
        print_section("RUNNING PARTITION COUNT VALIDATION")
        partition_count_result = validate_partition_count(source_df, target_df)
        print(partition_count_result)
        validation_results.append(partition_count_result)

        # 3) Schema & datatype drift (raw df okay; actual inferred types)
        print_section("RUNNING SCHEMA & DATATYPE DRIFT VALIDATION")
        schema_drift_result, schema_drift_detail_df = validate_schema_and_datatype_drift(
            source_df, target_df, column_mappings
        )
        print(schema_drift_result)
        validation_results.append(schema_drift_result)

        print("\nSCHEMA & DATATYPE DRIFT DETAILS:")
        print(schema_drift_detail_df)

        # 4) Duplicate key validation (uses normalized PK internally, raw PK in output)
        print_section("RUNNING DUPLICATE KEY VALIDATION")
        source_dup_result, source_dup_df = validate_duplicate_keys(source_work_df, primary_key_columns, "SOURCE")
        target_dup_result, target_dup_df = validate_duplicate_keys(target_work_df, primary_key_columns, "TARGET")

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

        # 5) Missing / Extra by key (uses normalized PK internally, raw PK in output)
        print_section("RUNNING MISSING / EXTRA BY KEY VALIDATION")
        missing_extra_result, extra_records_in_source_df, extra_records_in_target_df, matched_keys_df = validate_missing_and_extra_by_key(
            source_work_df, target_work_df, primary_key_columns
        )

        print(missing_extra_result)
        validation_results.append(missing_extra_result)

        if missing_extra_result["extra_records_in_source_count"] > 0:
            print("\nEXTRA RECORDS IN SOURCE (present in source, absent in target):")
            extra_records_in_source_df.show(truncate=False)

        if missing_extra_result["extra_records_in_target_count"] > 0:
            print("\nEXTRA RECORDS IN TARGET (present in target, absent in source):")
            extra_records_in_target_df.show(truncate=False)

        # 6) Hash mismatch validation (normalized joins + raw values in report + row-id-safe lookup)
        print_section("RUNNING HASH MISMATCH VALIDATION")
        hash_mismatch_result, detailed_mismatch_df = validate_hash_mismatch(
            source_df=source_df,
            target_df=target_df,
            source_work_df=source_work_df,
            target_work_df=target_work_df,
            primary_key_columns=primary_key_columns,
            compare_columns=compare_columns
        )

        print(hash_mismatch_result)
        validation_results.append(hash_mismatch_result)

        if hash_mismatch_result["mismatch_record_count"] > 0:
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
        # WRITE DETAILED OUTPUT FILES
        # ========================================================
        print_section("WRITING DETAILED OUTPUT FILES (PRODUCTION SAFE)")

        detailed_output_paths = write_detailed_output_files(
            local_tmp_dir=local_tmp_dir,
            s3_run_output_dir=run_output_dir,
            schema_drift_detail_df=schema_drift_detail_df,
            detailed_mismatch_spark_df=detailed_mismatch_df,
            extra_records_in_source_spark_df=extra_records_in_source_df,
            extra_records_in_target_spark_df=extra_records_in_target_df,
            source_dup_df=source_dup_df,
            target_dup_df=target_dup_df
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
            mismatch_record_count=hash_mismatch_result["mismatch_record_count"],
            mismatch_field_count=hash_mismatch_result["mismatch_field_count"],
            extra_records_in_source=missing_extra_result["extra_records_in_source_count"],
            extra_records_in_target=missing_extra_result["extra_records_in_target_count"],
            row_count_status=row_count_result["status"],
            partition_count_status=partition_count_result["status"],
            schema_drift_status=schema_drift_result["status"],
            schema_drift_total_columns=schema_drift_result["total_columns_checked"],
            schema_fail_columns=schema_drift_result["total_fail_columns"],
            schema_warn_columns=schema_drift_result["total_warn_columns"],
            source_duplicate_keys=source_dup_result["duplicate_key_count"],
            target_duplicate_keys=target_dup_result["duplicate_key_count"],
            overall_status=overall_status,
            schema_drift_detail_df=schema_drift_detail_df
        )

        print(f"Excel report created in S3: {excel_file_path}")

        # ========================================================
        # WRITE MANIFEST JSON
        # ========================================================
        print_section("WRITING MANIFEST JSON")

        manifest_path = write_manifest_json(
            s3_run_output_dir=run_output_dir,
            run_id=RUN_ID,
            source_table=SOURCE_PATH,
            target_table=TARGET_PATH,
            mapping_path=MAPPING_PATH,
            overall_status=overall_status,
            validation_summary=summary,
            detailed_output_paths=detailed_output_paths,
            excel_report_path=excel_file_path
        )

        print(f"Manifest JSON created in S3: {manifest_path}")

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