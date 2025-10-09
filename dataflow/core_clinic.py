import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from google.cloud import storage
import pandas as pd


def clean_parquet_to_dicts(parquet_path, sample_size=100):
    # Read parquet from GCS into pandas
    df = pd.read_parquet(parquet_path)

    # one preprocessing step: fillna with 0
    df.fillna(0, inplace=True)

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
    prefix = "clinic_table/"
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
        job_name="bq-clinic-table",
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
                table="dosewise-473716:dosewisedb.clinic_table",
                schema={
                    "fields": [
                        {"name": "caseid", "type": "INTEGER", "mode": "REQUIRED"},
                        {"name": "subjectid", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "casestart", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "caseend", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "anestart", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "aneend", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "opstart", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "opend", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "adm", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "dis", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "icu_days", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "death_inhosp", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "age", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "sex", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "height", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "weight", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "bmi", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "asa", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "emop", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "department", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "optype", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "dx", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "opname", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "approach", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "position", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "ane_type", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "preop_htn", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "preop_dm", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "preop_ecg", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "preop_pft", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "preop_hb", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_plt", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_pt", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_aptt", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_na", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_k", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_gluc", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_alb", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_ast", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_alt", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_bun", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_cr", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_ph", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_hco3", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_be", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_pao2", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_paco2", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "preop_sao2", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "cormack", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "airway", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "tubesize", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "dltubesize", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "lmasize", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "iv1", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "iv2", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "aline1", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "aline2", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "cline1", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "cline2", "type": "STRING", "mode": "NULLABLE"},
                        {"name": "intraop_ebl", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "intraop_uo", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "intraop_rbc", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_ffp", "type": "INTEGER", "mode": "NULLABLE"},
                        {
                            "name": "intraop_crystalloid",
                            "type": "FLOAT",
                            "mode": "NULLABLE",
                        },
                        {
                            "name": "intraop_colloid",
                            "type": "INTEGER",
                            "mode": "NULLABLE",
                        },
                        {"name": "intraop_ppf", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_mdz", "type": "FLOAT", "mode": "NULLABLE"},
                        {"name": "intraop_ftn", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_rocu", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_vecu", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_eph", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_phe", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_epi", "type": "INTEGER", "mode": "NULLABLE"},
                        {"name": "intraop_ca", "type": "INTEGER", "mode": "NULLABLE"},
                    ]
                },
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )


if __name__ == "__main__":
    run()
