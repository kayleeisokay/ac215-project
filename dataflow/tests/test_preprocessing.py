import apache_beam as beam
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions


class CleanData(beam.DoFn):
    def process(self, element):
        # element is a dict (one row)
        yield element


def clean_parquet_to_dicts(parquet_path):
    # Read parquet from GCS into pandas
    df = pd.read_parquet(parquet_path)

    # Imputation per id
    df["ART"] = df.groupby("id")["ART"].transform(lambda x: x.ffill().bfill())
    df["ECG_II"] = df.groupby("id")["ECG_II"].transform(lambda x: x.ffill().bfill())
    df["PLETH"] = df.groupby("id")["PLETH"].transform(lambda x: x.ffill().bfill())
    df["CO2"] = df.groupby("id")["CO2"].transform(lambda x: x.ffill().bfill())
    df["PHEN_RATE"] = df.groupby("id")["PHEN_RATE"].transform(lambda x: x.fillna(0))

    # Convert to list of dicts for Beam
    df = df.head(100)

    return df.to_dict(orient="records")


def run():
    options = PipelineOptions(
        runner="DirectRunner",  # Run locally first
        project="dosewise-473716",
        temp_location="gs://dosewisedb/tmp",
        region="us-central1",
    )

    with beam.Pipeline(options=options) as p:
        (
            p
            | "Read Parquet"
            >> beam.io.ReadFromParquet(
                "gs://dosewisedb/hemodyn_table/2025-10-07-09-16-52/hemodyn_table.parquet"
            )
            | "Clean Data" >> beam.ParDo(CleanData())
        )


if __name__ == "__main__":
    run()
