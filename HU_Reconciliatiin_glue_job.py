import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext

from pyspark.sql.functions import lit


# --------------------------------------------------
# Parameters
# --------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    [
        "SOURCE_PATH",
        "TARGET_PATH",
        "REPORT_PATH"
    ]
)

source_path = args["SOURCE_PATH"]
target_path = args["TARGET_PATH"]
report_path = args["REPORT_PATH"]


# --------------------------------------------------
# Initialize Spark
# --------------------------------------------------

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session


# --------------------------------------------------
# Read Source and Target
# --------------------------------------------------

print("Reading SOURCE...")
source_df = (
    spark.read
    .option("header", "true")
    .csv(source_path)
)

print("Reading TARGET...")
target_df = (
    spark.read
    .option("header", "true")
    .csv(target_path)
)


# --------------------------------------------------
# Compare customer IDs
# --------------------------------------------------

source_ids = source_df.select("customer_id").distinct()

target_ids = target_df.select("customer_id").distinct()


missing_in_target = source_ids.subtract(target_ids)

extra_in_target = target_ids.subtract(source_ids)


# --------------------------------------------------
# Generate reconciliation report
# --------------------------------------------------

missing_report = (
    missing_in_target
    .withColumn(
        "issue",
        lit("MISSING_IN_TARGET")
    )
)

extra_report = (
    extra_in_target
    .withColumn(
        "issue",
        lit("EXTRA_IN_TARGET")
    )
)

reconciliation_report = (
    missing_report
    .unionByName(extra_report)
)


print("================================")
print("RECONCILIATION REPORT")
print("================================")

reconciliation_report.show()


# --------------------------------------------------
# Write reconciliation report
# --------------------------------------------------

reconciliation_report.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(report_path)


print("Reconciliation report written to:")
print(report_path)


# --------------------------------------------------
# CORRECT TARGET
#
# Source is the master dataset.
# Therefore the corrected target becomes
# identical to Source.
# --------------------------------------------------

print("================================")
print("CORRECTING TARGET DATA")
print("================================")

(
    source_df
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(target_path)
)


print("================================")
print("TARGET CORRECTION COMPLETED")
print("================================")

print("Target is now synchronized with Source.")
