import pandas as pd
import requests
from io import StringIO
from google.cloud import storage
import tempfile
from datetime import datetime

BUCKET_NAME = "dosewisedb"


def export_lab_table_to_gcs(request):
    # URL for the lab cases
    url = "https://api.vitaldb.net/labs"

    # Fetch the CSV
    resp = requests.get(url)
    resp.raise_for_status()

    # Decode CSV text
    csv_text = resp.content.decode("utf-8-sig")

    # Read CSV into DataFrame
    labs_df = pd.read_csv(StringIO(csv_text))

    # Optional: filter to relevant cases if you want
    # Assuming you have a list of case IDs
    import vitaldb

    features = ["ART", "ECG_II", "PLETH", "CO2", "PHEN_RATE"]
    cases = vitaldb.find_cases(features)
    lab_subset = labs_df[labs_df["caseid"].isin(cases)]

    # Generate timestamp for folder
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    gcs_path = f"lab_table/{timestamp}/lab_table.parquet"

    # Save to temp file and upload to GCS
    with tempfile.NamedTemporaryFile() as tmpfile:
        lab_subset.to_parquet(tmpfile.name, index=False)

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(tmpfile.name)

    return f"Uploaded {len(lab_subset)} rows to gs://{BUCKET_NAME}/{gcs_path}", 200
