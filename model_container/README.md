# Model Container

This container is designed to pull and process data from Google Cloud Storage (GCS) for the AC215 DoseWise project.

## Prerequisites

- Docker installed on your machine ([Install Docker](https://docs.docker.com/get-docker/))
- GCP service account credentials with access to the `dosewise` project
- SSH access configured for GitHub (if working with the repository)

## Setup Instructions

### 1. Get GCP Service Account Credentials

You need a GCP service account key to access the data bucket:

== Most users will not need to do this, since we have already set up the service account credentials in the secrets folder. Request .json file from Kaylee, Chloe, or Adrian. ==

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select the `dosewise-473716` project
3. Navigate to **IAM & Admin** → **Service Accounts**
4. Select the appropriate service account (or create one with Storage Object Viewer permissions)
5. Click **Keys** → **Add Key** → **Create New Key**
6. Choose **JSON** format and download the key
7. Save the downloaded JSON file as `dosewise-473716-9f4874e812d6.json`

### 2. Place Credentials in the Secrets Folder

Create a `secrets/` directory in the project root (if it doesn't exist) and place your credentials file there:

```bash
# From the project root directory
mkdir -p secrets
mv ~/Downloads/dosewise-473716-9f4874e812d6.json secrets/
```

**Important:** The `secrets/` folder is already in `.gitignore` to prevent accidentally committing credentials to git.

### 3. Build the Docker Image

Navigate to the `model_container` directory and build the image:

```bash
cd model_container
docker build -t baseline-model -f Dockerfile .
```

This will:
- Set up a Python 3.11 environment
- Install all required dependencies (pandas, gcsfs, google-cloud-storage, pyarrow)
- Copy the application code into the container

### 4. Run the Container and Pull Data

Run the container with the credentials mounted:

```bash
docker run --rm \
  -v /path/to/project/secrets/dosewise-473716-9f4874e812d6.json:/app/dosewise-473716-9f4874e812d6.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/dosewise-473716-9f4874e812d6.json \
  baseline-model -lc "source /home/app/.venv/bin/activate && python main.py"
```

**Note:** Replace `/path/to/project/` with the absolute path to your project directory.

**For convenience, you can use the provided script:**

```bash
# Edit docker.sh to update the path to your secrets folder
./docker.sh
```

### Expected Output

When successful, you should see output like:

```
Loading latest file: hemodyn_table/2025-10-04-09-16-36/hemodyn_table.parquet
Successfully loaded data with shape: (1866721, 7)

First few rows:
    id      ART    ECG_II      PLETH  CO2  PHEN_RATE  time
0  513      NaN       NaN        NaN  NaN        NaN     1
1  513  5.01471  0.622872  30.839600  0.0        NaN     2
...
```

## What the Container Does

The `main.py` script automatically:
1. Connects to Google Cloud Storage using the provided credentials
2. Lists all hemodyn_table parquet files in the `dosewisedb` bucket
3. Identifies and loads the **most recent** timestamped file
4. Displays basic information about the loaded dataset

No need to manually update timestamps - the script always pulls the latest data!

## Available Data Tables

The `dosewisedb` bucket contains three main tables:
- `clinic_table/` - Clinical data
- `hemodyn_table/` - Hemodynamic measurements (currently loaded)
- `lab_table/` - Laboratory results

## Troubleshooting

### Permission Denied Errors

If you get permission errors:
- Verify your service account has the **Storage Object Viewer** role. If not, reach out to Kaylee, Chloe, or Adrian.
- Check that the credentials file path is correct
- Ensure the credentials file is valid (not expired)

### File Not Found

If the script can't find data files:
- Verify you have access to the `dosewisedb` bucket
- Check that data has been uploaded to the bucket
- Confirm your GCP project ID is correct

### Docker Build Issues

If the build fails:
- Ensure Docker is running
- Check your internet connection (needed to download dependencies)
- Try clearing Docker cache: `docker system prune`

## Modifying the Container

### Change Which Table to Load

Edit `main.py` and update the `table_prefix` variable:

```python
table_prefix = "clinic_table/"  # or "lab_table/"
```

### Add New Dependencies

1. Add the package to `pyproject.toml`:
   ```toml
   dependencies = [
       "package-name>=version",
   ]
   ```
2. Rebuild the Docker image

## Project Structure

```
model_container/
├── Dockerfile          # Docker image definition
├── pyproject.toml      # Python dependencies
├── main.py            # Data loading script
├── docker.sh          # Convenience script to build and run
└── README.md          # This file
```

## Next Steps

After successfully pulling the data, you can:
- Perform exploratory data analysis (EDA)
- Build predictive models
- Process and transform the data
- Export results for further analysis

## Support

For questions or issues, contact the team or refer to the main project README.

