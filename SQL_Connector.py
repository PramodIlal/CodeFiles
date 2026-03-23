import os
import boto3
import pandas as pd
from sqlalchemy import create_engine

# =========================
# CONFIGURATION
# =========================

MYSQL_HOST = "YOUR_MYSQL_HOST"
MYSQL_PORT = 3306
MYSQL_DB = "etl_db"
MYSQL_USER = "etl_user"
MYSQL_PASSWORD = "StrongPassword123!"

AWS_REGION = "ap-south-1"
S3_BUCKET = "your-etl-bucket"

SOURCE_TABLE = "source_orders"
TARGET_TABLE = "target_orders"

# =========================
# MYSQL CONNECTION
# =========================

mysql_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(mysql_url)

# =========================
# S3 CLIENT
# =========================

s3 = boto3.client("s3", region_name=AWS_REGION)

# =========================
# HELPER FUNCTION
# =========================

def extract_and_upload(table_name, zone):
    print(f"\n[INFO] Extracting table: {table_name}")

    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)

    print(f"[INFO] Rows fetched: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")

    local_file = f"{table_name}.csv"
    df.to_csv(local_file, index=False)

    s3_key = f"raw/{zone}/{table_name}/data.csv"
    s3.upload_file(local_file, S3_BUCKET, s3_key)

    print(f"[SUCCESS] Uploaded to s3://{S3_BUCKET}/{s3_key}")

    os.remove(local_file)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    extract_and_upload(SOURCE_TABLE, "source")
    extract_and_upload(TARGET_TABLE, "target")
    print("\n[DONE] Source and target tables uploaded to S3 successfully.")
