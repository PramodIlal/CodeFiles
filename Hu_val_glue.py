import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext


# --------------------------------------------------
# Get parameters
# --------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    [
        "SOURCE_PATH",
        "TARGET_PATH"
    ]
)

source_path = args["SOURCE_PATH"]
target_path = args["TARGET_PATH"]


# --------------------------------------------------
# Initialize Glue / Spark
# --------------------------------------------------

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session


# --------------------------------------------------
# Read Source and Target
# --------------------------------------------------

print("Reading SOURCE data...")
source_df = (
    spark.read
    .option("header", "true")
    .csv(source_path)
)

print("Reading TARGET data...")
target_df = (
    spark.read
    .option("header", "true")
    .csv(target_path)
)


# --------------------------------------------------
# Record count validation
# --------------------------------------------------

source_count = source_df.count()
target_count = target_df.count()

print("================================")
print("SOURCE COUNT:", source_count)
print("TARGET COUNT:", target_count)
print("================================")


# --------------------------------------------------
# Customer ID comparison
# --------------------------------------------------

source_ids = source_df.select("customer_id").distinct()

target_ids = target_df.select("customer_id").distinct()


# IDs present in Source but missing from Target
missing_in_target = source_ids.subtract(target_ids)


# IDs present in Target but not Source
extra_in_target = target_ids.subtract(source_ids)


missing_count = missing_in_target.count()
extra_count = extra_in_target.count()


print("Missing in Target:", missing_count)
print("Extra in Target:", extra_count)


# --------------------------------------------------
# Validation result
# --------------------------------------------------

if (
    source_count == target_count
    and missing_count == 0
    and extra_count == 0
):

    print("================================")
    print("VALIDATION PASSED")
    print("================================")

else:

    print("================================")
    print("VALIDATION FAILED")
    print("================================")

    print("Records missing in Target:")
    missing_in_target.show()

    print("Records extra in Target:")
    extra_in_target.show()

    # Make Glue job fail so Step Functions
    # can catch the failure.
    raise Exception(
        "Source and Target validation failed"
    )
