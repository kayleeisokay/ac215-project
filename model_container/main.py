def main():
    import pandas as pd

    bucket_name = "dosewisedb"
    file_path_in_bucket = "hemodyn_table.csv"

    # Directly read from GCS
    df = pd.read_csv(f"gs://{bucket_name}/{file_path_in_bucket}")

    print(df.head())


if __name__ == "__main__":
    main()
