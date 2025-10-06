

def main():
    import pandas as pd
    from google.cloud import storage

    bucket_name = "dosewisedb"
    table_prefix = "hemodyn_table/"
    
    # Initialize GCS client and get the bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List all blobs in the hemodyn_table directory
    blobs = list(bucket.list_blobs(prefix=table_prefix))
    
    # Filter for parquet files and sort by name (timestamps are in the path)
    parquet_files = [blob.name for blob in blobs if blob.name.endswith('.parquet')]
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {table_prefix}")
    
    # Sort to get the latest (timestamps are chronologically sortable)
    latest_file = sorted(parquet_files)[-1]
    
    print(f"Loading latest file: {latest_file}")
    
    # Directly read from GCS
    df = pd.read_parquet(f"gs://{bucket_name}/{latest_file}")

    print(f"Successfully loaded data with shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
