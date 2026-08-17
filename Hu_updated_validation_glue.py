import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext


# ============================================================
# 1. GET PARAMETERS
# ============================================================

args = getResolvedOptions(
    sys.argv,
    [
        "SOURCE_PATH",
        "TARGET_PATH"
    ]
)

source_path = args["SOURCE_PATH"]
target_path = args["TARGET_PATH"]


# ============================================================
# 2. INITIALIZE GLUE / SPARK
# ============================================================

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session


# ============================================================
# 3. READ SOURCE
# ============================================================

print("========================================")
print("READING SOURCE DATA")
print("========================================")

source_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(source_path)
)


# ============================================================
# 4. READ TARGET
# ============================================================

print("========================================")
print("READING TARGET DATA")
print("========================================")

target_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(target_path)
)


# ============================================================
# 5. RECORD COUNT VALIDATION
# ============================================================

source_count = source_df.count()
target_count = target_df.count()

print("========================================")
print("RECORD COUNT VALIDATION")
print("========================================")

print(f"Source record count : {source_count}")
print(f"Target record count : {target_count}")


# ============================================================
# 6. CUSTOMER ID COMPARISON
# ============================================================

source_ids = (
    source_df
    .select("customer_id")
    .distinct()
)

target_ids = (
    target_df
    .select("customer_id")
    .distinct()
)


# IDs present in Source but missing from Target
extra_in_source = source_ids.subtract(target_ids)


# IDs present in Target but missing from Source
extra_in_target = target_ids.subtract(source_ids)


extra_in_source_count = extra_in_source.count()
extra_in_target_count = extra_in_target.count()


print("========================================")
print("DATA COMPARISON")
print("========================================")

print(
    f"Extra records in Source : "
    f"{extra_in_source_count}"
)

print(
    f"Extra records in Target : "
    f"{extra_in_target_count}"
)


# ============================================================
# 7. VALIDATION RESULT
# ============================================================

if (
    source_count == target_count
    and extra_in_source_count == 0
    and extra_in_target_count == 0
):

    print("========================================")
    print("VALIDATION PASSED")
    print("========================================")

else:

    print("========================================")
    print("VALIDATION FAILED")
    print("========================================")

    print("Extra in Source:")
    extra_in_source.show()

    print("Extra in Target:")
    extra_in_target.show()

    raise Exception(
        "Source and Target validation failed"
    )
