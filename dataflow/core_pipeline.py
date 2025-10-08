import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from google.cloud import storage
import pandas as pd


def clean_parquet_to_dicts(parquet_path, sample_size=100):
    # Read parquet from GCS into pandas
    df = pd.read_parquet(parquet_path)

    # Imputation per id
    df["ART"] = df.groupby("id")["ART"].transform(lambda x: x.ffill().bfill())
    df["ECG_II"] = df.groupby("id")["ECG_II"].transform(lambda x: x.ffill().bfill())
    df["PLETH"] = df.groupby("id")["PLETH"].transform(lambda x: x.ffill().bfill())
    df["CO2"] = df.groupby("id")["CO2"].transform(lambda x: x.ffill().bfill())
    df["PHEN_RATE"] = df.groupby("id")["PHEN_RATE"].transform(lambda x: x.fillna(0))

    # Optional: sort by time
    df = df.sort_values("time")

    return df.to_dict(orient="records")


def get_latest_parquet_path(bucket_name, prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    parquet_files = [b.name for b in blobs if b.name.endswith(".parquet")]
    latest = sorted(parquet_files)[-1]
    return f"gs://{bucket_name}/{latest}"


def run():
    bucket_name = "dosewisedb"
    prefix = "hemodyn_table/"
    latest_parquet_path = get_latest_parquet_path(bucket_name, prefix)
    print("Using latest parquet file:", latest_parquet_path)

    # Clean a small sample locally
    cleaned_records = clean_parquet_to_dicts(latest_parquet_path, sample_size=100)

    options = PipelineOptions(
        runner="DataflowRunner",
        project="dosewise-473716",
        region="us-central1",
        temp_location="gs://dosewisedb/tmp",
        staging_location="gs://dosewisedb/staging",
        job_name="clean-data-sample",
        save_main_session=True,
        experiments=["use_runner_v2"],
        sdk_container_image="apache/beam_python3.9_sdk:2.58.0",
        flink_version="1.18",
    )

    options.view_as(SetupOptions).requirements_file = "requirements.txt"

    with beam.Pipeline(options=options) as p:
        (
            p
            | "Create PCollection" >> beam.Create(cleaned_records)
            | "Write to BigQuery"
            >> beam.io.WriteToBigQuery(
                table="dosewise-473716:dosewisedb.hemodyn_table",
                schema={
                    "fields": [
                        {"name": "id", "type": "INTEGER", "mode": "REQUIRED"},
                        {"name": "ART", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "ECG_II", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "PLETH", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "CO2", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "PHEN_RATE", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "time", "type": "INTEGER", "mode": "REQUIRED"},
                    ]
                },
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )


if __name__ == "__main__":
    run()
