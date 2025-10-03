import pandas as pd
import requests
from io import StringIO
from google.cloud import storage
import tempfile
import vitaldb
from datetime import datetime

BUCKET_NAME = "dosewisedb"


def export_clinic_table_to_gcs(request):
    features = ["ART", "ECG_II", "PLETH", "CO2", "PHEN_RATE"]
    cases = vitaldb.find_cases(features)

    # Fetch clinical cases CSV from API
    url = "https://api.vitaldb.net/cases"
    resp = requests.get(url)
    resp.raise_for_status()

    csv_text = resp.content.decode("utf-8-sig")
    cases_df = pd.read_csv(StringIO(csv_text))
    filtered_df = cases_df[cases_df["caseid"].isin(cases)]

    # Generate timestamp
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # GCS path with timestamp folder
    gcs_path = f"clinic_table/{timestamp}/clinic_table.parquet"

    # Save to temp file and upload
    with tempfile.NamedTemporaryFile() as tmpfile:
        filtered_df.to_parquet(tmpfile.name, index=False)

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(tmpfile.name)

    return (
        f"Uploaded {len(filtered_df)} rows to gs://{BUCKET_NAME}/{gcs_path}",
        200,
    )
