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
