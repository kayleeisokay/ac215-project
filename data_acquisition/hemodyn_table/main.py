import vitaldb
import pandas as pd
from google.cloud import storage
import tempfile
from tqdm import tqdm
from datetime import datetime

BUCKET_NAME = "dosewisedb"


def export_hemodyn_to_gcs(request):
    features = ["ART", "ECG_II", "PLETH", "CO2", "PHEN_RATE"]
    cases = vitaldb.find_cases(features)
    print("Number of cases found:", len(cases))

    values_for_all_cases = {}
    for case in tqdm(cases):
        vals = vitaldb.load_case(case, features, interval=1)
        values_for_all_cases[case] = vals

    rows = []
    for k, arr in values_for_all_cases.items():
        for row in arr:
            rows.append([k] + list(row))

    df = pd.DataFrame(rows, columns=["id"] + features)
    df["time"] = range(1, len(df) + 1)

    # Generate timestamp
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # GCS path with timestamp folder
    gcs_path = f"hemodyn_table/{timestamp}/hemodyn_table.parquet"

    # Save to temporary file and upload
    with tempfile.NamedTemporaryFile() as tmpfile:
        df.to_parquet(tmpfile.name, index=False)

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(tmpfile.name)

    return f"Uploaded {len(cases)} cases to gs://{BUCKET_NAME}/{gcs_path}", 200
