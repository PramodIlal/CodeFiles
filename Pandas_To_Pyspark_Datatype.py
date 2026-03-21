from decimal import Decimal
def convert_pandas_to_pyspark_dtype(pandas_dtype: str, sample_data: pd.Series = None) -> Dict[str, str]:
    """Convert pandas dtype to PySpark data type format."""
    dtype_lower = str(pandas_dtype).lower()

    # ------------------------------------------------------------
    # ADDITION: Handle DynamoDB import-from-S3 string values first
    # (minimal required enhancement for your project)
    # ------------------------------------------------------------
    if sample_data is not None:
        non_null = sample_data.dropna()

        if not non_null.empty:
            sample_vals = non_null.head(10).tolist()
            first_val = sample_vals[0]

            # Future-safe: if DynamoDB number comes as Decimal in some tables
            if isinstance(first_val, Decimal):
                has_fraction = any(
                    isinstance(v, Decimal) and v != v.to_integral_value()
                    for v in sample_vals
                )
                return {"data_type": "double" if has_fraction else "integer"}

            # Handle string values (your current DynamoDB import-from-S3 case)
            if isinstance(first_val, str):
                values = [str(v).strip() for v in sample_vals if v is not None]

                # Integer-like strings
                if values and all(re.match(r"^-?\d+$", v) for v in values):
                    return {"data_type": "integer"}

                # Float-like strings
                if values and all(re.match(r"^-?\d+(\.\d+)?$", v) for v in values):
                    return {"data_type": "double"}

                # Boolean-like strings
                lower_vals = {v.lower() for v in values}
                if lower_vals.issubset({"true", "false"}):
                    return {"data_type": "boolean"}

                # Timestamp format: YYYY-MM-DD HH:mm:ss
                if re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(:\d{2})?$', values[0]):
                    return {"data_type": "timestamp", "date_format": "yyyy-MM-dd HH:mm:ss"}

                # Timestamp format: M/D/YYYY HH:mm:ss or MM/DD/YYYY HH:mm:ss
                if re.match(r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(:\d{2})?$', values[0]):
                    return {"data_type": "timestamp", "date_format": "M/d/yyyy HH:mm:ss"}

                # Date format: YYYY-MM-DD
                if re.match(r'^\d{4}-\d{2}-\d{2}$', values[0]):
                    try:
                        pd.to_datetime(values[:5], format='%Y-%m-%d', errors='raise')
                        return {"data_type": "date", "date_format": "yyyy-MM-dd"}
                    except:
                        pass

                # Date format: MM/DD/YYYY or M/D/YYYY
                if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', values[0]):
                    try:
                        pd.to_datetime(values[:5], format='%m/%d/%Y', errors='raise')
                        return {"data_type": "date", "date_format": "M/d/yyyy"}
                    except:
                        pass

    # ------------------------------------------------------------
    # EXISTING LOGIC (kept as-is)
    # ------------------------------------------------------------
    if sample_data is not None and pd.api.types.is_datetime64_any_dtype(sample_data):
        first_val = sample_data.dropna().iloc[0] if not sample_data.dropna().empty else None
        if first_val:
            date_str = str(first_val)
            if '/' in date_str:
                return {"data_type": "date", "date_format": "M/d/yyyy"}
            elif '-' in date_str:
                return {"data_type": "date", "date_format": "yyyy-MM-dd"}
        return {"data_type": "date"}

    if sample_data is not None and dtype_lower in ('object', 'string'):
        first_val = sample_data.dropna().iloc[0] if not sample_data.dropna().empty else None
        if first_val:
            date_str = str(first_val).strip()

            if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', date_str):
                try:
                    pd.to_datetime(sample_data.head(5), format='%m/%d/%Y', errors='coerce')
                    return {"data_type": "date", "date_format": "M/d/yyyy"}
                except:
                    pass

            elif re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
                try:
                    pd.to_datetime(sample_data.head(5), format='%Y-%m-%d', errors='coerce')
                    return {"data_type": "date", "date_format": "yyyy-MM-dd"}
                except:
                    pass

            elif re.match(r'^\d{2}-\d{2}-\d{4}', date_str):
                try:
                    pd.to_datetime(sample_data.head(5), format='%d-%m-%Y', errors='coerce')
                    return {"data_type": "date", "date_format": "dd-MM-yyyy"}
                except:
                    pass

            elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', date_str):
                return {"data_type": "timestamp", "date_format": "M/d/yyyy HH:mm:ss"}

            elif re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}', date_str):
                return {"data_type": "timestamp", "date_format": "yyyy-MM-dd HH:mm:ss"}

    if 'bool' in dtype_lower:
        return {"data_type": "boolean"}

    if 'int64' in dtype_lower:
        return {"data_type": "integer"}
    elif 'int32' in dtype_lower or 'int' in dtype_lower:
        return {"data_type": "integer"}

    if 'float64' in dtype_lower:
        return {"data_type": "double"}
    elif 'float32' in dtype_lower:
        return {"data_type": "double"}

    return {"data_type": "string"}
