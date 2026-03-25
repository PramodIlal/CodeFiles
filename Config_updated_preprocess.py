import sys
import json
import boto3
from urllib.parse import urlparse

from pyspark.context import SparkContext
from pyspark.sql.functions import col
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

# ===============================
# READ ARGUMENTS
# ===============================
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'CONFIG_S3_PATH'])

config_s3_path = args['CONFIG_S3_PATH']

# ===============================
# GLUE / SPARK CONTEXT
# ===============================
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.sparkContext.setLogLevel("ERROR")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

s3_client = boto3.client("s3")

# ===============================
# READ JSON CONFIG FROM S3
# ===============================
def read_json_from_s3(s3_path):
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    return json.loads(content)

config = read_json_from_s3(config_s3_path)

# ===============================
# NEW: READ MASTER CONFIG SECTIONS
# ===============================
workflow_type = config.get("workflow_type")
multi_source_config = config.get("multi_source_workflow", {})

if workflow_type != "multi_source":
    print(f"Skipping preprocessing: This job supports only workflow_type='multi_source', but got '{workflow_type}'")
    job.commit()
    sys.exit(0)

# ===============================
# PARSE S3 PATH
# ===============================
def parse_s3_path(s3_path):
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip('/')

# ===============================
# READ DATA FUNCTION
# ===============================
def read_data(source_config):
    source_type = source_config["type"]

    if source_type == "s3":
        path = source_config["path"]

        if path.endswith(".csv"):
            return spark.read.option("header", True).csv(path)

        elif path.endswith(".json"):
            return spark.read.json(path)

        elif path.endswith(".parquet"):
            return spark.read.parquet(path)

        else:
            raise ValueError(f"Unsupported S3 file format for path: {path}")

    elif source_type == "mysql":
        url = f"jdbc:mysql://{source_config['host']}:{source_config['port']}/{source_config['database']}"
        return spark.read.format("jdbc") \
            .option("url", url) \
            .option("dbtable", source_config["table"]) \
            .option("user", source_config["username"]) \
            .option("password", source_config["password"]) \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .load()

    else:
        raise ValueError(f"Unsupported source type: {source_type}")

# ===============================
# FIND PART FILE IN TEMP FOLDER
# ===============================
def find_parquet_part_file(bucket, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # pick actual parquet part file only
            if key.endswith(".parquet") and "/part-" in key:
                return key

    raise Exception(f"No parquet part file found in s3://{bucket}/{prefix}")

# ===============================
# DELETE S3 PREFIX (FOLDER)
# ===============================
def delete_s3_prefix(bucket, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")

    keys_to_delete = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys_to_delete.append({"Key": obj["Key"]})

            # delete in batches of 1000
            if len(keys_to_delete) == 1000:
                s3_client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": keys_to_delete}
                )
                keys_to_delete = []

    if keys_to_delete:
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": keys_to_delete}
        )

# ===============================
# WRITE SINGLE PARQUET FILE WITH FIXED NAME
# ===============================
def write_single_parquet_file(df, final_s3_file_path):
    """
    Writes dataframe as a single parquet file with fixed name in S3.
    Example final_s3_file_path:
      s3://bucket/raw/processed_source/new_source.parquet
    """

    bucket, final_key = parse_s3_path(final_s3_file_path)

    # temp folder path
    temp_key = final_key.replace(".parquet", "_temp/")
    temp_s3_path = f"s3://{bucket}/{temp_key}"

    # 1) write single partition to temp folder
    df.coalesce(1).write.mode("overwrite").parquet(temp_s3_path)

    # 2) find generated part file
    part_file_key = find_parquet_part_file(bucket, temp_key)

    # 3) if final file already exists, delete it first
    try:
        s3_client.head_object(Bucket=bucket, Key=final_key)
        s3_client.delete_object(Bucket=bucket, Key=final_key)
        print(f"Deleted existing file: s3://{bucket}/{final_key}")
    except s3_client.exceptions.ClientError:
        pass

    # 4) copy part file to final fixed filename
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": part_file_key},
        Key=final_key
    )

    print(f"Copied {part_file_key} -> {final_key}")

    # 5) delete temp folder
    delete_s3_prefix(bucket, temp_key)
    print(f"Deleted temp folder: s3://{bucket}/{temp_key}")

# ===============================
# STEP 1: READ ONLY REQUIRED TABLES
# ===============================
required_tables = set()

for step in multi_source_config["join_steps"]:
    if step["left"] != "result":
        required_tables.add(step["left"])
    required_tables.add(step["right"])

source_dfs = {}

for src in multi_source_config["sources"]:
    if src["name"] in required_tables:
        df = read_data(src)
        source_dfs[src["name"]] = df.alias(src["name"])

print(f"Loaded source tables: {list(source_dfs.keys())}")

# ===============================
# STEP 2: APPLY JOINS
# ===============================
joined_df = None
primary_table = None
for i, step in enumerate(multi_source_config["join_steps"]):
    print(f"Join {i+1}: {step['left']} ← {step['type']} → {step['right']}")
   
    if i == 0:
        # First join: use the actual table DataFrames
        left_df = source_dfs[step["left"]]
        primary_table = step["left"]  # Track the base table
        left_key_col = col(f"{step['left']}.{step['left_key']}")
    else:
        # Subsequent joins: use the result of previous join
        left_df = joined_df
       
        # For 'result' references, use the primary table name
        if step["left"] == "result":
            # The join key should reference the original table where it came from
            # In most cases, this is the first table (primary_table)
            left_key_col = col(f"{primary_table}.{step['left_key']}")
        else:
            left_key_col = col(f"{step['left']}.{step['left_key']}")
   
    right_df = source_dfs[step["right"]]
    right_key_col = col(f"{step['right']}.{step['right_key']}")
   
    # Perform the join
    joined_df = left_df.join(
        right_df,
        left_key_col == right_key_col,
        step.get("type", "inner")
    )
 

print("Joins completed successfully")

# ===============================
# STEP 3: SELECT + RENAME SOURCE COLUMNS
# ===============================
mapped_columns = multi_source_config["column_mapping"]

source_cols = [
    col(src_col).alias(mapped_columns[src_col])
    for src_col in multi_source_config["select_columns"]
]

processed_source = joined_df.select(*source_cols)

print("Processed source schema:")
processed_source.printSchema()

# ===============================
# STEP 4: READ TARGET
# ===============================
target_df = read_data(multi_source_config["target"])

# ===============================
# STEP 5: FILTER TARGET COLUMNS
# ===============================
target_cols = list(mapped_columns.values())
processed_target = target_df.select(*target_cols)

print("Processed target schema:")
processed_target.printSchema()

# ===============================
# STEP 6: WRITE OUTPUTS WITH FIXED FILE NAMES
# ===============================
# IMPORTANT:
# output.source and output.target should be FOLDER paths in config
# Example:
# "source": "s3://your-etl-bucket/raw/processed_source/"
# "target": "s3://your-etl-bucket/raw/processed_target/"

source_output_folder = multi_source_config["output"]["source"].rstrip("/")
target_output_folder = multi_source_config["output"]["target"].rstrip("/")

final_source_file = f"{source_output_folder}/new_source.parquet"
final_target_file = f"{target_output_folder}/new_target.parquet"

write_single_parquet_file(processed_source, final_source_file)
write_single_parquet_file(processed_target, final_target_file)

print(f"Processed source written to: {final_source_file}")
print(f"Processed target written to: {final_target_file}")

print("✅ Preprocessing completed successfully")

job.commit()
