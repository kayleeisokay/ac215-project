import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


opts = PipelineOptions(
    [
        "--runner=DataflowRunner",
        "--project=dosewise-473716",
        "--region=us-central1",
        "--temp_location=gs://dosewisedb/tmp",
        "--staging_location=gs://dosewisedb/staging",
        "--flink_version=1.18",  # override to a supported version
        "--experiments=use_runner_v2",  # recommended for modern SDK images
    ]
)


with beam.Pipeline(options=opts) as p:
    (
        p
        | "Create" >> beam.Create([1, 2, 3])
        | "Map" >> beam.Map(lambda x: x * 10)
        | "Print" >> beam.Map(print)
    )
