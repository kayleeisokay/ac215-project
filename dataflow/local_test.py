import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


class CleanData(beam.DoFn):
    def process(self, element):
        import pandas as pd

        df = pd.DataFrame([element])
        df["ART"] = df.groupby("id")["ART"].transform(lambda x: x.ffill().bfill())
        df["ECG_II"] = df.groupby("id")["ECG_II"].transform(lambda x: x.ffill().bfill())
        df["PLETH"] = df.groupby("id")["PLETH"].transform(lambda x: x.ffill().bfill())
        df["CO2"] = df.groupby("id")["CO2"].transform(lambda x: x.ffill().bfill())
        df["PHEN_RATE"] = df.groupby("id")["PHEN_RATE"].transform(lambda x: x.fillna(0))
        for _, row in df.iterrows():
            yield row.to_dict()


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
            # | "Preview" >> beam.Map(print)
        )


if __name__ == "__main__":
    run()
