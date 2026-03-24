import sys
import json
import boto3
from urllib.parse import urlparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from awsglue.utils import getResolvedOptions

# ===============================
# READ ARGUMENTS
# ===============================
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'CONFIG_S3_PATH'])

config_s3_path = args['CONFIG_S3_PATH']

spark = SparkSession.builder.appName("Preprocessing Job").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

s3_client = boto3.client("s3")

# ===============================
# READ CONFIG FROM S3
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
# STEP 1: READ ONLY REQUIRED TABLES
# ===============================
required_tables = set()

for step in config["join_steps"]:
    if step["left"] != "result":
        required_tables.add(step["left"])
    required_tables.add(step["right"])

source_dfs = {}

for src in config["sources"]:
    if src["name"] in required_tables:
        df = read_data(src)
        source_dfs[src["name"]] = df.alias(src["name"])

print(f"Loaded source tables: {list(source_dfs.keys())}")

# ===============================
# STEP 2: APPLY JOINS
# ===============================
joined_df = None

for i, step in enumerate(config["join_steps"]):
    left_name = step["left"]
    right_name = step["right"]
    left_key = step["left_key"]
    right_key = step["right_key"]
    join_type = step.get("type", "inner")

    if i == 0:
        left_df = source_dfs[left_name]
    else:
        left_df = joined_df

    right_df = source_dfs[right_name]

    joined_df = left_df.join(
        right_df,
        col(f"{left_name}.{left_key}") == col(f"{right_name}.{right_key}"),
        join_type
    ).alias("result")

print("Joins completed successfully")

# ===============================
# STEP 3: SELECT + RENAME SOURCE COLUMNS
# ===============================
mapped_columns = config["column_mapping"]

source_cols = [
    col(src_col).alias(mapped_columns[src_col])
    for src_col in config["select_columns"]
]

processed_source = joined_df.select(*source_cols)

print("Processed source schema:")
processed_source.printSchema()

# ===============================
# STEP 4: READ TARGET
# ===============================
target_df = read_data(config["target"])

# ===============================
# STEP 5: FILTER TARGET COLUMNS
# ===============================
target_cols = list(mapped_columns.values())

processed_target = target_df.select(*target_cols)

print("Processed target schema:")
processed_target.printSchema()

# ===============================
# STEP 6: WRITE OUTPUTS
# ===============================
source_output = config["output"]["source"]
target_output = config["output"]["target"]

processed_source.write.mode("overwrite").parquet(source_output)
processed_target.write.mode("overwrite").parquet(target_output)

print(f"Processed source written to: {source_output}")
print(f"Processed target written to: {target_output}")

print("✅ Preprocessing completed successfully")
