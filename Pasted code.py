#this file uses schema-at-read-time for efficient data type enforcement

import json
import os
from datetime import datetime
from urllib.parse import urlparse

import boto3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, ArrayType, MapType, 
    DateType, TimestampType, StringType, IntegerType, 
    DoubleType, BooleanType
)

# AWS Glue imports
import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job


# ============================================================
# CONFIGURATION (SUPPORTS MULTIPLE FILE FORMATS)
# ============================================================
SOURCE_PATH = "data/E-commerce-Source.csv"           # Can be .csv, .json, .parquet, .xlsx OR dynamodb:table_name
TARGET_PATH = "data/E-commerce-Target copy.csv"      # Can be .csv, .json, .parquet, .xlsx  OR dynamodb:table_name
MAPPING_PATH = "data/ecom_datatype_mapping.json"
OUTPUT_BASE_PATH = "output"

# Optional: Format-specific configurations
FILE_CONFIG = {
    "source": {
        "multiline": True,        # For JSON files
        "sheet_name": 0,          # For Excel files
        "merge_schema": False     # For Parquet files
    },
    "target": {
        "multiline": True,
        "sheet_name": 0,
        "merge_schema": False
    }
}


# ============================================================
# AWS / S3 / DYNAMODB HELPERS
# ============================================================
def is_s3_path(path):
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def upload_file_to_s3(local_path, s3_uri):
    """
    Upload local file to S3
    """
    if not os.path.exists(local_path):
        print(f"[WARN] File not found, skipping upload: {local_path}")
        return

    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    print(f"[OK] Uploaded to S3: {s3_uri}")


def upload_run_output_dir_to_s3(local_run_dir, s3_output_base):
    """
    Upload all generated output files from local run dir to S3.
    Example:
      local_run_dir = /tmp/validation_output/run_20260320_123000
      s3_output_base = s3://my-bucket/reports
    Uploads files to:
      s3://my-bucket/reports/run_20260320_123000/<file>
    """
    if not is_s3_path(s3_output_base):
        print("[INFO] OUTPUT_BASE_PATH is not an S3 path. Skipping S3 upload.")
        return

    run_folder_name = os.path.basename(local_run_dir.rstrip("/"))

    for file_name in os.listdir(local_run_dir):
        local_file = os.path.join(local_run_dir, file_name)
        if os.path.isfile(local_file):
            s3_dest = f"{s3_output_base.rstrip('/')}/{run_folder_name}/{file_name}"
            upload_file_to_s3(local_file, s3_dest)


def is_dynamodb_source(source_path):
    """
    Detects whether source is a DynamoDB table reference.
    Expected format:
      dynamodb:table_name
    Example:
      dynamodb:source_audit_table
    """
    return isinstance(source_path, str) and source_path.startswith("dynamodb:")


def extract_dynamodb_table_name(source_path):
    """
    Extract DynamoDB table name from:
      dynamodb:table_name
    """
    return source_path.split("dynamodb:", 1)[1].strip()


# ============================================================
# SPARK SESSION
# ============================================================
def create_spark_session(app_name="ETL Validation Job"):
    """
    IMPORTANT FOR GLUE:
    Do NOT use .master("local[*]") in AWS Glue.
    Glue manages Spark cluster itself.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ============================================================
# FILE FORMAT DETECTION
# ============================================================
def detect_file_format(file_path):
    """
    Detects file format based on extension.
    """
    extension = file_path.lower().split('.')[-1]
    format_map = {
        'csv': 'csv',
        'json': 'json',
        'parquet': 'parquet',
        'xlsx': 'excel',
        'xls': 'excel'
    }
    detected_format = format_map.get(extension, None)
    
    if detected_format is None:
        raise ValueError(f"Unsupported file extension: {extension}")
    
    return detected_format


def detect_source_format(source_path):
    """
    Detect source format for both file paths and DynamoDB source.
    """
    if is_dynamodb_source(source_path):
        return "dynamodb"
    return detect_file_format(source_path)


# ============================================================
# SCHEMA BUILDER FROM MAPPING
# ============================================================
def build_schema_from_mapping(column_mappings, dataset_name="source"):
    """
    Builds a PySpark StructType schema from mapping JSON.
    
    Args:
        column_mappings: List of column mapping dicts from JSON
        dataset_name: "source" or "target" - determines which column name to use
    
    Returns:
        StructType schema
    """
    type_map = {
        "string": StringType(),
        "integer": IntegerType(),
        "double": DoubleType(),
        "boolean": BooleanType(),
        "date": DateType(),
        "timestamp": TimestampType()
    }
    
    fields = []
    for mapping in column_mappings:
        # Choose column name based on dataset
        if dataset_name == "source":
            col_name = mapping["source_column"]
        else:
            col_name = mapping["target_column"]
        
        # Get data type from mapping, default to string if not specified
        data_type_str = mapping.get("data_type", "string")
        date_format = mapping.get("date_format")
        
        # KEY FIX: If date/timestamp has custom format, read as STRING first
        # We'll convert it to proper date/timestamp in apply_custom_date_formats()
        if data_type_str in ["date", "timestamp"] and date_format:
            # Read as string, will be converted later
            spark_type = StringType()
            print(f"[SCHEMA] {col_name}: Reading as STRING (will convert to {data_type_str} with format '{date_format}')")
        else:
            # Use normal type mapping
            spark_type = type_map.get(data_type_str, StringType())
        
        # All columns nullable by default
        fields.append(StructField(col_name, spark_type, True))
    
    return StructType(fields)


# ============================================================
# FILE READERS WITH SCHEMA SUPPORT
# ============================================================
def read_csv_with_schema(spark, file_path, schema):
    """
    Reads CSV with predefined schema (most efficient for CSV).
    """
    return (
        spark.read
        .schema(schema)
        .option("header", "true")
        .option("mode", "PERMISSIVE")  # Handle malformed records gracefully
        .csv(file_path)
    )


def read_json_with_schema(spark, file_path, schema=None, multiline=True):
    """
    Reads JSON file with optional schema enforcement.
    """
    reader = spark.read.option("multiLine", str(multiline).lower())
    
    if schema:
        reader = reader.schema(schema)
    else:
        reader = reader.option("inferSchema", "true")
    
    return reader.json(file_path)


def read_parquet_file(spark, file_path, merge_schema=False):
    """
    Reads Parquet file (uses embedded schema).
    """
    return (
        spark.read
        .option("mergeSchema", str(merge_schema).lower())
        .parquet(file_path)
    )


def read_excel_with_schema(spark, file_path, schema=None, sheet_name=0):
    """
    Reads Excel file using pandas, then converts to Spark DataFrame.
    Applies schema by casting columns if provided.
    """
    # Read Excel with pandas
    pandas_df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)
    
    # Apply schema by casting columns if provided
    if schema:
        for field in schema.fields:
            if field.name in spark_df.columns:
                spark_df = spark_df.withColumn(
                    field.name,
                    F.col(field.name).cast(field.dataType)
                )
    
    return spark_df


def read_dynamodb_table(glueContext, table_name, schema=None):
    """
    Reads DynamoDB table into Spark DataFrame.
    If schema (StructType) is provided, applies column selection and casting after read.
    """
    print(f"\n[INFO] Reading DynamoDB table: {table_name}")

    dyf = glueContext.create_dynamic_frame.from_options(
        connection_type="dynamodb",
        connection_options={
            "dynamodb.input.tableName": table_name,
            "dynamodb.throughput.read.percent": "0.5"
        }
    )

    df = dyf.toDF()

    if schema:
        existing_cols = set(df.columns)

        select_exprs = []
        missing_cols = []

        for field in schema.fields:
            if field.name in existing_cols:
                select_exprs.append(
                    col(field.name).cast(field.dataType.simpleString()).alias(field.name)
                )
            else:
                missing_cols.append(field.name)

        if missing_cols:
            print(f"[WARN] Missing expected columns in DynamoDB source: {missing_cols}")

        if select_exprs:
            df = df.select(*select_exprs)

    return df


def apply_custom_date_formats(df, column_mappings, dataset_name):
    """
    Applies custom date/timestamp format conversions after schema enforcement.
    This is needed when your mapping has custom date_format specifications.
    
    Args:
        df: Spark DataFrame
        column_mappings: List of column mapping dicts
        dataset_name: "source" or "target"
    
    Returns:
        DataFrame with custom date formats applied
    """
    for mapping in column_mappings:
        col_name = (mapping["source_column"] if dataset_name == "source" 
                   else mapping["target_column"])
        
        data_type = mapping.get("data_type")
        date_format = mapping.get("date_format")
        
        # Only process if column exists in DataFrame
        if col_name not in df.columns:
            continue
        
        # Handle custom date format conversion
        if data_type == "date" and date_format:
            df = df.withColumn(
                col_name,
                F.when(
                    F.col(col_name).isNull() | (F.trim(F.col(col_name).cast("string")) == ""),
                    F.lit(None).cast("date")
                ).otherwise(
                    F.to_date(F.col(col_name), date_format)
                )
            )
        
        elif data_type == "timestamp" and date_format:
            df = df.withColumn(
                col_name,
                F.when(
                    F.col(col_name).isNull() | (F.trim(F.col(col_name).cast("string")) == ""),
                    F.lit(None).cast("timestamp")
                ).otherwise(
                    F.to_timestamp(F.col(col_name), date_format)
                )
            )
    
    return df


def read_file_smart(spark, file_path, column_mappings, dataset_name, **kwargs):
    """
    Smart file reader that chooses the best approach based on format.
    
    - CSV: Uses schema-at-read-time (most efficient)
    - JSON: Uses schema-at-read-time with custom date handling
    - Parquet: Uses embedded schema
    - Excel: Reads then applies schema
    
    Args:
        spark: SparkSession
        file_path: Path to file
        column_mappings: Column mappings from JSON
        dataset_name: "source" or "target"
        **kwargs: Format-specific options
    
    Returns:
        Spark DataFrame with enforced schema
    """
    file_format = detect_file_format(file_path)
    schema = build_schema_from_mapping(column_mappings, dataset_name)
    
    print(f"\n[INFO] Reading {dataset_name.upper()} file: {file_path}")
    print(f"[INFO] File format: {file_format.upper()}")
    print(f"[INFO] Using predefined schema from mapping JSON")
    
    # CSV: Most efficient with schema-at-read
    if file_format == 'csv':
        df = read_csv_with_schema(spark, file_path, schema)
        print_section("check")
        df.show()
        # Apply custom date formats if needed
        df = apply_custom_date_formats(df, column_mappings, dataset_name)
        return df
    
    # JSON: Schema-at-read with multiline support
    elif file_format == 'json':
        multiline = kwargs.get('multiline', True)
        df = read_json_with_schema(spark, file_path, schema, multiline)
        # Apply custom date formats if needed
        df = apply_custom_date_formats(df, column_mappings, dataset_name)
        return df
    
    # Parquet: Uses embedded schema
    elif file_format == 'parquet':
        merge_schema = kwargs.get('merge_schema', False)
        df = read_parquet_file(spark, file_path, merge_schema)
        # Cast columns to match expected schema
        df = cast_to_expected_schema(df, schema)
        return df
    
    # Excel: Read with pandas then apply schema
    elif file_format == 'excel':
        sheet_name = kwargs.get('sheet_name', 0)
        df = read_excel_with_schema(spark, file_path, schema, sheet_name)
        # Apply custom date formats if needed
        df = apply_custom_date_formats(df, column_mappings, dataset_name)
        return df
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def read_source_smart(glueContext, spark, source_path, column_mappings, dataset_name, **kwargs):
    """
    Reads source either from:
      - DynamoDB table (dynamodb:table_name)
      - File path (csv/json/parquet/xlsx)

    For DynamoDB:
      - uses build_schema_from_mapping(..., dataset_name='source')
      - applies custom schema after reading

    For file:
      - uses existing read_file_smart() unchanged
    """
    # SOURCE from DynamoDB
    if is_dynamodb_source(source_path):
        table_name = extract_dynamodb_table_name(source_path)
        source_schema = build_schema_from_mapping(column_mappings, dataset_name="source")
        df = read_dynamodb_table(glueContext, table_name, schema=source_schema)

        # Apply custom date/timestamp conversions if needed
        df = apply_custom_date_formats(df, column_mappings, dataset_name="source")

        return df

    # SOURCE from file (existing logic)
    return read_file_smart(
        spark,
        source_path,
        column_mappings=column_mappings,
        dataset_name=dataset_name,
        **kwargs
    )


def read_target_smart(glueContext, spark, target_path, column_mappings, dataset_name, **kwargs):
    """
    Reads target either from:
      - DynamoDB table (dynamodb:table_name)
      - File path (csv/json/parquet/xlsx)

    For DynamoDB:
      - uses build_schema_from_mapping(..., dataset_name='target')
      - applies custom schema after reading

    For file:
      - uses existing read_file_smart() unchanged
    """
    # TARGET from DynamoDB
    if is_dynamodb_source(target_path):   # reuse existing prefix checker
        table_name = extract_dynamodb_table_name(target_path)   # reuse existing extractor
        target_schema = build_schema_from_mapping(column_mappings, dataset_name="target")
        df = read_dynamodb_table(glueContext, table_name, schema=target_schema)

        # Apply custom date/timestamp conversions if needed
        df = apply_custom_date_formats(df, column_mappings, dataset_name="target")

        return df

    # TARGET from file (existing logic unchanged)
    return read_file_smart(
        spark,
        target_path,
        column_mappings=column_mappings,
        dataset_name=dataset_name,
        **kwargs
    )

def cast_to_expected_schema(df, expected_schema):
    """
    Casts DataFrame columns to match expected schema.
    Used for Parquet files which have embedded schemas.
    
    Args:
        df: Spark DataFrame
        expected_schema: StructType schema to match
    
    Returns:
        DataFrame with casted columns
    """
    for field in expected_schema.fields:
        if field.name in df.columns:
            df = df.withColumn(field.name, F.col(field.name).cast(field.dataType))
    
    return df


def read_mapping_json(mapping_path):
    """
    Reads mapping JSON from local file system or S3.
    """
    if is_s3_path(mapping_path):
        bucket, key = parse_s3_uri(mapping_path)
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=key)
        mapping = json.loads(response["Body"].read().decode("utf-8"))
    else:
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

    primary_keys = mapping.get("primary_keys", [])
    column_mappings = mapping.get("column_mappings", [])

    if not primary_keys:
        raise ValueError("Mapping JSON must contain 'primary_keys'.")

    if not column_mappings:
        raise ValueError("Mapping JSON must contain 'column_mappings'.")

    # Validate data type specifications (if provided)
    valid_data_types = ["string", "integer", "double", "date", "timestamp", "boolean"]
    
    for mapping_entry in column_mappings:
        data_type = mapping_entry.get("data_type")
        
        if data_type:
            # Validate data type
            if data_type not in valid_data_types:
                raise ValueError(
                    f"Invalid data_type '{data_type}' for column '{mapping_entry.get('target_column')}'. "
                    f"Valid types: {valid_data_types}"
                )
            
            # Validate date_format for date/timestamp types
            if data_type in ["date", "timestamp"]:
                if not mapping_entry.get("date_format"):
                    raise ValueError(
                        f"Column '{mapping_entry.get('target_column')}' has data_type '{data_type}' "
                        f"but missing 'date_format' specification."
                    )

    return primary_keys, column_mappings


# ============================================================
# COMPLEX TYPE HANDLING
# ============================================================
def has_nested_structures(df):
    """
    Checks if DataFrame has nested struct/array/map columns.
    """
    for field in df.schema.fields:
        if isinstance(field.dataType, (StructType, ArrayType, MapType)):
            return True
    return False


def flatten_dataframe(df, delimiter="_", max_depth=5):
    """
    Flattens nested struct columns recursively.
    
    Args:
        df: Spark DataFrame
        delimiter: Separator for flattened column names
        max_depth: Maximum nesting depth to flatten
    """
    def flatten_struct(df, prefix="", depth=0):
        if depth >= max_depth:
            return df
        
        flat_cols = []
        has_struct = False
        
        for field in df.schema.fields:
            col_name = field.name
            col_type = field.dataType
            
            full_col_name = f"{prefix}{delimiter}{col_name}" if prefix else col_name
            
            if isinstance(col_type, StructType):
                has_struct = True
                # Expand struct fields
                for nested_field in col_type.fields:
                    nested_col_name = f"{full_col_name}{delimiter}{nested_field.name}"
                    flat_cols.append(
                        F.col(f"{col_name}.{nested_field.name}").alias(nested_col_name)
                    )
            elif isinstance(col_type, ArrayType):
                # Convert arrays to JSON strings
                flat_cols.append(F.to_json(F.col(col_name)).alias(full_col_name))
            elif isinstance(col_type, MapType):
                # Convert maps to JSON strings
                flat_cols.append(F.to_json(F.col(col_name)).alias(full_col_name))
            else:
                # Simple types
                flat_cols.append(F.col(col_name).alias(full_col_name))
        
        df_flat = df.select(*flat_cols)
        
        # Recursively flatten if there are still struct columns
        if has_struct and depth < max_depth:
            return flatten_struct(df_flat, prefix="", depth=depth + 1)
        
        return df_flat
    
    return flatten_struct(df)


def normalize_data_types_for_comparison(df, strict=False):
    """
    Normalizes data types for cross-format comparison.
    
    Args:
        strict: If False, convert dates/timestamps to strings for comparison
    """
    if strict:
        return df
    
    # Convert dates and timestamps to strings for consistent comparison
    normalized_df = df
    for field in df.schema.fields:
        if isinstance(field.dataType, DateType):
            normalized_df = normalized_df.withColumn(
                field.name,
                F.date_format(F.col(field.name), "yyyy-MM-dd")
            )
        elif isinstance(field.dataType, TimestampType):
            normalized_df = normalized_df.withColumn(
                field.name,
                F.date_format(F.col(field.name), "yyyy-MM-dd HH:mm:ss")
            )
    
    return normalized_df


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
    # run_output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    run_output_dir = os.path.join(base_output_dir, "run_validation")
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
    Enhanced version: Handles complex types (struct, array, map).
    Normalizes compare columns before hashing/comparison.
    """
    schema_map = {field.name: field.dataType for field in df.schema.fields}
    
    normalized_df = df
    for col_name in compare_columns:
        col_type = schema_map.get(col_name)
        
        # Handle complex types
        if isinstance(col_type, (StructType, ArrayType, MapType)):
            # Convert complex type to JSON string for comparison
            normalized_df = normalized_df.withColumn(
                col_name,
                F.when(F.col(col_name).isNull(), F.lit(""))
                 .otherwise(F.to_json(F.col(col_name)))
            )
        else:
            # Simple types - existing logic
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


def get_schema_dict(df, include_complex_details=False):
    """
    Returns schema as dict: {column_name: data_type_string}
    """
    if not include_complex_details:
        return {field.name: field.dataType.simpleString() for field in df.schema.fields}
    
    schema_info = {}
    for field in df.schema.fields:
        if isinstance(field.dataType, (StructType, ArrayType, MapType)):
            # For complex types, include full structure as JSON
            schema_info[field.name] = str(field.dataType.jsonValue())
        else:
            schema_info[field.name] = field.dataType.simpleString()
    
    return schema_info


def spark_df_to_pandas_safe(df):
    """
    Converts Spark DF to Pandas DF safely for small/medium result sets.
    For this project, output detail datasets are expected to be manageable.
    """
    if df is None:
        return pd.DataFrame()
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


def write_detailed_output_files(
    run_output_dir,
    schema_drift_detail_df,
    detailed_mismatch_spark_df,
    extra_records_in_source_spark_df,
    extra_records_in_target_spark_df
):
    """
    Writes detailed output files as CSV.
    """
    # 1) Schema drift CSV
    schema_drift_csv_path = os.path.join(run_output_dir, "schema_datatype_drift.csv")
    write_csv_from_pandas(schema_drift_detail_df, schema_drift_csv_path)

    # 2) Detailed mismatch CSV
    mismatch_pdf = spark_df_to_pandas_safe(detailed_mismatch_spark_df)
    mismatch_csv_path = os.path.join(run_output_dir, "detailed_mismatch.csv")
    write_csv_from_pandas(mismatch_pdf, mismatch_csv_path)

    # 3) Extra records in source CSV
    extra_source_pdf = spark_df_to_pandas_safe(extra_records_in_source_spark_df)
    extra_source_csv_path = os.path.join(run_output_dir, "extra_records_in_source.csv")
    write_csv_from_pandas(extra_source_pdf, extra_source_csv_path)

    # 4) Extra records in target CSV
    extra_target_pdf = spark_df_to_pandas_safe(extra_records_in_target_spark_df)
    extra_target_csv_path = os.path.join(run_output_dir, "extra_records_in_target.csv")
    write_csv_from_pandas(extra_target_pdf, extra_target_csv_path)

    return {
        "schema_drift_csv": schema_drift_csv_path,
        "detailed_mismatch_csv": mismatch_csv_path,
        "extra_records_in_source_csv": extra_source_csv_path,
        "extra_records_in_target_csv": extra_target_csv_path
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
    schema_drift_status,
    schema_drift_total_columns,
    schema_drift_columns,
    schema_dtype_mismatch_count,
    source_duplicate_keys,
    target_duplicate_keys,
    overall_status,
    schema_drift_detail_df,
    detailed_mismatch_spark_df,
    extra_records_in_source_spark_df,
    extra_records_in_target_spark_df,
    source_format,
    target_format
):
    """
    Generates one Excel file with multiple sheets:
      - summary_report
      - schema_datatype_drift
      - detailed_mismatch
      - extra_in_source
      - extra_in_target
    """
    file_name = "validation_report.xlsx"
    output_path = os.path.join(run_output_dir, file_name)

    # Summary sheet
    summary_data = [{
        "source_format": source_format,
        "target_format": target_format,
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
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }]
    summary_df = pd.DataFrame(summary_data)

    # Convert Spark DFs to Pandas for Excel sheets
    detailed_mismatch_pdf = spark_df_to_pandas_safe(detailed_mismatch_spark_df)
    extra_source_pdf = spark_df_to_pandas_safe(extra_records_in_source_spark_df)
    extra_target_pdf = spark_df_to_pandas_safe(extra_records_in_target_spark_df)

    # Write workbook
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_report", index=False)
        schema_drift_detail_df.to_excel(writer, sheet_name="schema_datatype_drift", index=False)
        detailed_mismatch_pdf.to_excel(writer, sheet_name="detailed_mismatch", index=False)
        extra_source_pdf.to_excel(writer, sheet_name="extra_in_source", index=False)
        extra_target_pdf.to_excel(writer, sheet_name="extra_in_target", index=False)

    print(f"[OK] Excel report generated: {output_path}")
    return output_path


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
# MAIN (GLUE READY + DYNAMODB SOURCE SUPPORT)
# ============================================================
def main():
    global SOURCE_PATH, TARGET_PATH, MAPPING_PATH, OUTPUT_BASE_PATH

    # ============================================================
    # GLUE JOB PARAMETER HANDLING
    # ============================================================
    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "SOURCE_PATH",
            "TARGET_PATH",
            "MAPPING_PATH",
            "OUTPUT_BASE_PATH"
        ]
    )

    # Override hardcoded paths at runtime using Glue parameters
    SOURCE_PATH = args["SOURCE_PATH"]
    TARGET_PATH = args["TARGET_PATH"]
    MAPPING_PATH = args["MAPPING_PATH"]
    OUTPUT_BASE_PATH = args["OUTPUT_BASE_PATH"]

    print_section("GLUE JOB PARAMETERS")
    print(f"SOURCE_PATH      : {SOURCE_PATH}")
    print(f"TARGET_PATH      : {TARGET_PATH}")
    print(f"MAPPING_PATH     : {MAPPING_PATH}")
    print(f"OUTPUT_BASE_PATH : {OUTPUT_BASE_PATH}")

    # ============================================================
    # GLUE CONTEXT + JOB INIT
    # ============================================================
    spark = create_spark_session()
    glueContext = GlueContext(spark.sparkContext)
    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)

    # ============================================================
    # LOCAL TEMP OUTPUT DIR FOR GLUE EXECUTION
    # If user passes S3 output path, write locally first, then upload.
    # ============================================================
    actual_output_base_path = OUTPUT_BASE_PATH
    if is_s3_path(OUTPUT_BASE_PATH):
        actual_output_base_path = "/tmp/validation_output"
        os.makedirs(actual_output_base_path, exist_ok=True)
        print(f"[INFO] Using local temp output dir for generation: {actual_output_base_path}")

    # Create timestamped run folder
    run_output_dir = create_run_output_dir(actual_output_base_path)

    print_section("RUN OUTPUT DIRECTORY")
    print(f"Run output directory: {run_output_dir}")

    print_section("READING MAPPING JSON")
    primary_keys, column_mappings = read_mapping_json(MAPPING_PATH)

    # Detect file formats
    source_format = detect_source_format(SOURCE_PATH)
    target_format = detect_source_format(TARGET_PATH)

    print_section("BUILDING SCHEMAS FROM MAPPING")
    source_schema = build_schema_from_mapping(column_mappings, dataset_name="source")
    target_schema = build_schema_from_mapping(column_mappings, dataset_name="target")

    print("Source schema from mapping:")
    for field in source_schema.fields:
        print(f"  {field.name}: {field.dataType.simpleString()}")

    print("\nTarget schema from mapping:")
    for field in target_schema.fields:
        print(f"  {field.name}: {field.dataType.simpleString()}")

    # ========================================================
    # READ FILES / SOURCE WITH SCHEMA ENFORCEMENT
    # ========================================================
    print_section("READING FILES WITH SCHEMA ENFORCEMENT")

    # Read source with schema (supports file OR DynamoDB)
    source_df = read_source_smart(
        glueContext,
        spark,
        SOURCE_PATH,
        column_mappings=column_mappings,
        dataset_name="source",
        **FILE_CONFIG["source"]
    )

    # Read target with schema (existing file logic)
    target_df = read_target_smart(
        glueContext,
        spark,
        TARGET_PATH,
        column_mappings=column_mappings,
        dataset_name="target",
        **FILE_CONFIG["target"]
    )

    print("\nSource schema (after read with enforcement):")
    source_df.printSchema()

    print("\nTarget schema (after read with enforcement):")
    target_df.printSchema()

    print("Source columns:", source_df.columns)
    print("Target columns:", target_df.columns)
    print_section("source")
    source_df.show()

    # ========================================================
    # PRE-PROCESSING FOR FORMAT-SPECIFIC HANDLING
    # ========================================================
    print_section("PRE-PROCESSING FOR FILE FORMATS")

    # Handle JSON/Parquet nested structures (if any remain after schema enforcement)
    if has_nested_structures(source_df):
        print(f"[INFO] Source has nested structures. Flattening...")
        source_df = flatten_dataframe(source_df)
        print(f"Source columns (after flattening): {source_df.columns}")

    if has_nested_structures(target_df):
        print(f"[INFO] Target has nested structures. Flattening...")
        target_df = flatten_dataframe(target_df)
        print(f"Target columns (after flattening): {target_df.columns}")

    # Normalize data types for cross-format comparison (if needed)
    if source_format != target_format:
        print(f"[INFO] Source format ({source_format}) differs from target format ({target_format}).")
        print(f"[INFO] Normalizing data types for cross-format comparison...")
        source_df = normalize_data_types_for_comparison(source_df, strict=False)
        target_df = normalize_data_types_for_comparison(target_df, strict=False)

    # ========================================================
    # 1) PRE-MAPPING COLUMN VALIDATION
    # ========================================================
    source_required_columns = [
        m["source_column"] for m in column_mappings
        if m["source_column"] is not None
    ]
    target_required_columns = [m["target_column"] for m in column_mappings]

    print_section("PRE-MAPPING COLUMN VALIDATION")
    source_missing_cols = validate_required_columns(source_df, source_required_columns, "SOURCE")
    target_missing_cols = validate_required_columns(target_df, target_required_columns, "TARGET")

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
    print_section("WRITING DETAILED OUTPUT CSV FILES")

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
    # GENERATE EXCEL REPORT (MULTIPLE SHEETS)
    # ========================================================
    print_section("GENERATING EXCEL REPORT")

    excel_file_path = generate_excel_report(
        run_output_dir=run_output_dir,
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
        schema_drift_detail_df=schema_drift_detail_df,
        detailed_mismatch_spark_df=detailed_mismatch_df,
        extra_records_in_source_spark_df=extra_records_in_source_df,
        extra_records_in_target_spark_df=extra_records_in_target_df,
        source_format=source_format,
        target_format=target_format
    )

    print(f"Excel report created: {excel_file_path}")

    # ========================================================
    # UPLOAD GENERATED OUTPUTS TO S3 IF OUTPUT_BASE_PATH IS S3
    # ========================================================
    if is_s3_path(OUTPUT_BASE_PATH):
        print_section("UPLOADING OUTPUT FILES TO S3")
        upload_run_output_dir_to_s3(run_output_dir, OUTPUT_BASE_PATH)

    # ========================================================
    # FINAL SUMMARY (CONSOLE)
    # ========================================================
    print_summary(overall_status, summary)

    # Stop Spark
    spark.stop()

    # Commit Glue job
    job.commit()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()