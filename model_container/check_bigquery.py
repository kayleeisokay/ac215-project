"""
Check available BigQuery datasets and tables
"""

from google.cloud import bigquery
import os

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/yseo/ac215-project-main/secrets/dosewise-473716-9f4874e812d6.json'

client = bigquery.Client(project='dosewise-473716')

print("="*60)
print("Checking BigQuery Datasets and Tables")
print("="*60)

# List all datasets
print("\n✓ Available Datasets:")
try:
    datasets = list(client.list_datasets())
    if datasets:
        for dataset in datasets:
            # Get full dataset details
            dataset_full = client.get_dataset(dataset.dataset_id)
            print(f"  - {dataset.dataset_id} (location: {dataset_full.location})")
            
            # List tables in each dataset
            tables = list(client.list_tables(dataset.dataset_id))
            if tables:
                print(f"    Tables:")
                for table in tables:
                    table_ref = client.get_table(f"{dataset.dataset_id}.{table.table_id}")
                    print(f"      • {table.table_id} ({table_ref.num_rows:,} rows, {table_ref.num_bytes:,} bytes)")
            else:
                print(f"    No tables found")
    else:
        print("  No datasets found in project")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)

