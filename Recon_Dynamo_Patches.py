def is_dynamodb_path(path):
    """
    Detects whether a path is a DynamoDB table reference.
    Expected format:
      dynamodb:table_name
    """
    return isinstance(path, str) and path.startswith("dynamodb:")



def extract_dynamodb_table_name(path):
    """
    Extract DynamoDB table name from:
      dynamodb:table_name
    """
    return path.split("dynamodb:", 1)[1].strip()




from pyspark.sql.functions import col

def read_dynamodb_table(glueContext, table_name, schema=None):
    """
    Reads DynamoDB table into Spark DataFrame.
    If schema is provided, applies column selection and casting after read.
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
            print(f"[WARN] Missing expected columns in DynamoDB table: {missing_cols}")

        if select_exprs:
            df = df.select(*select_exprs)

    return df



def read_target_smart(glueContext, spark, target_path, column_mappings, dataset_name, **kwargs):
    """
    Reads target either from:
      - DynamoDB table (dynamodb:table_name)
      - File path (csv/json/parquet/xlsx)

    For DynamoDB:
      - uses build_schema_from_mapping(..., dataset_name='target')
      - applies custom date/timestamp conversions after read

    For file:
      - uses existing read_file_smart() unchanged
    """
    if is_dynamodb_path(target_path):
        table_name = extract_dynamodb_table_name(target_path)

        target_schema = build_schema_from_mapping(column_mappings, dataset_name="target")

        df = read_dynamodb_table(glueContext, table_name, schema=target_schema)

        # Only if your reconciliation script already has this function
        df = apply_custom_date_formats(df, column_mappings, dataset_name="target")

        return df

    # Existing file logic unchanged
    return read_file_smart(
        spark,
        target_path,
        column_mappings=column_mappings,
        dataset_name=dataset_name,
        **kwargs
    )


def detect_input_format(path):
    """
    Detects input format for both files and DynamoDB references.
    """
    if is_dynamodb_path(path):
        return "dynamodb"
    return detect_file_format(path)


from awsglue.dynamicframe import DynamicFrame



def write_dynamodb_table(glueContext, df, table_name):
    """
    Writes Spark DataFrame to DynamoDB table.
    NOTE:
      - Best used when schema/column names already match target table
      - For full replace, DynamoDB does NOT support overwrite like files
      - This performs item upserts based on table keys
    """
    print(f"\n[INFO] Writing reconciled output to DynamoDB table: {table_name}")

    dyf = DynamicFrame.fromDF(df, glueContext, "reconciled_dyf")

    glueContext.write_dynamic_frame.from_options(
        frame=dyf,
        connection_type="dynamodb",
        connection_options={
            "dynamodb.output.tableName": table_name,
            "dynamodb.throughput.write.percent": "0.5"
        }
    )

    print(f"[OK] Reconciled output written to DynamoDB: {table_name}")






def write_target_smart(glueContext, df, target_path, file_write_func=None, **kwargs):
    """
    Writes reconciled output either to:
      - DynamoDB table (dynamodb:table_name)
      - File path using existing file write logic
    """
    if is_dynamodb_path(target_path):
        table_name = extract_dynamodb_table_name(target_path)
        write_dynamodb_table(glueContext, df, table_name)
        return

    # Existing file write logic unchanged
    if file_write_func is not None:
        file_write_func(df, target_path, **kwargs)
    else:
        raise ValueError("No file write function provided for non-DynamoDB target.")





target_format = detect_input_format(TARGET_PATH)




target_df = read_target_smart(
    glueContext,
    spark,
    TARGET_PATH,
    column_mappings=column_mappings,
    dataset_name="target",
    **FILE_CONFIG["target"]
)

write_target_smart(
    glueContext,
    reconciled_df,
    TARGET_PATH,
    file_write_func=write_reconciled_output
)




def write_reconciled_file_output(df, output_path):
    """
    Existing file write logic for reconciled output.
    Adjust format logic if needed.
    """
    df.write.mode("overwrite").option("header", "true").csv(output_path)



def read_source_smart(glueContext, spark, source_path, column_mappings, dataset_name, **kwargs):
    if is_dynamodb_path(source_path):
        table_name = extract_dynamodb_table_name(source_path)

        source_schema = build_schema_from_mapping(column_mappings, dataset_name="source")

        df = read_dynamodb_table(glueContext, table_name, schema=source_schema)

        df = apply_custom_date_formats(df, column_mappings, dataset_name="source")

        return df

    return read_file_smart(
        spark,
        source_path,
        column_mappings=column_mappings,
        dataset_name=dataset_name,
        **kwargs
    )


