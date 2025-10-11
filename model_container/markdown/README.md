# DoseWise Model Container

This container implements a counterfactual time series prediction model that predicts patient vital signs (arterial blood pressure) with and without drug intervention. The model uses an encoder-decoder architecture with Transformer and Conv1D layers.

## Model Architecture

The model consists of:
- **Encoder**: Conv1D layers → Positional Encoding → Transformer Encoder
- **Dual Decoders**: 
  - Control Branch (u=0): Predicts outcomes without drug intervention
  - Treatment Branch (u=dose): Predicts outcomes with drug intervention

### Features
- Input: ABP (arterial pressure), ECG_II, PLETH (plethysmography), CO2 (capnography), PHEN_RATE (phenylephrine infusion rate)
- Output: Future ABP predictions for both control and treatment scenarios

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

### 4. Train the Model

The model pulls data directly from BigQuery and trains on patient vital signs.

**Run training:**

```bash
docker run --rm \
  -v /path/to/project/secrets/dosewise-473716-9f4874e812d6.json:/app/dosewise-473716-9f4874e812d6.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/dosewise-473716-9f4874e812d6.json \
  -v $(pwd)/models:/app/models \
  baseline-model python main.py --train --num_epochs 10 --batch_size 32
```

**Training Options:**

```bash
# Train with custom hyperparameters
python main.py \
  --train \
  --batch_size 32 \
  --num_epochs 10 \
  --learning_rate 0.001 \
  --d_model 128 \
  --nhead 8 \
  --num_encoder_layers 4 \
  --num_decoder_layers 4 \
  --sequence_length 100 \
  --prediction_length 50

# Evaluate only (requires pre-trained model)
python main.py --evaluate

# Both train and evaluate
python main.py --train --evaluate
```

**Note:** Replace `/path/to/project/` with the absolute path to your project directory.

### Expected Output

**During Training:**

```
Using device: cuda
Loading data from BigQuery: dosewise-473716.dosewise.hemodyn_table
Executing query...
Loaded 1866721 rows from BigQuery
Number of unique patients: 103

Train patients: 74
Val patients: 8
Test patients: 21

Creating model...
Model created with 2,456,832 parameters

Training on cuda...
Epoch 1/10: 100%|████████| 1234/1234 [02:15<00:00, 9.12it/s]
Epoch 1/10 - Train Loss: 0.3245, Val Loss: 0.2987
...
Model saved to dosewise_model.pth
```

**During Evaluation:**

```
Evaluating model on test set...
Test Loss (MSE): 0.2856
Test RMSE: 0.5344
```

## What the Container Does

The model training pipeline (`main.py`) automatically:
1. **Connects to BigQuery** using the provided credentials
2. **Pulls patient vital signs data** (ABP, ECG_II, PLETH, CO2, PHEN_RATE) from the `hemodyn_table`
3. **Splits data by patient ID** to prevent data leakage (train/val/test split)
4. **Normalizes features** using StandardScaler and saves the scaler for inference
5. **Creates time series sequences** with sliding windows for temporal modeling
6. **Trains the dual-decoder model** to predict counterfactual outcomes
7. **Evaluates performance** on held-out test patients
8. **Saves trained model** (`dosewise_model.pth`) and training artifacts

### Model Components

**`model.py`** contains:
- `PositionalEncoding`: Adds positional information to sequences
- `DoseWiseEncoder`: Encodes input vital signs into hidden representations
- `DoseWiseDecoder`: Decodes predictions from hidden states
- `DoseWiseModel`: Complete model with dual decoder branches
- `create_model()`: Factory function for easy model instantiation

**`main.py`** contains:
- `VitalSignsDataset`: PyTorch dataset for loading time series sequences
- `load_data_from_bigquery()`: Pulls data from BigQuery warehouse
- `normalize_data()`: Standardizes features for training
- `train_model()`: Training loop with validation
- `main()`: Complete training and evaluation pipeline

## Available Data Sources

The model pulls data from BigQuery tables in the `dosewise` dataset:
- `hemodyn_table` - Hemodynamic measurements (ABP, ECG_II, PLETH, CO2, PHEN_RATE) - **Currently used**
- `clinic_table` - Clinical data (patient demographics, diagnoses)
- `lab_table` - Laboratory results (blood work, etc.)

### Output Files

After training, the following files are generated:
- `dosewise_model.pth` - Trained model weights
- `scaler.pkl` - Feature normalization scaler (for inference)
- `training_history.pkl` - Training and validation loss history

## Troubleshooting

### Permission Denied Errors

If you get permission errors:
- Verify your service account has the **BigQuery Data Viewer** and **BigQuery Job User** roles
- Ensure the service account has **Storage Object Viewer** for model artifact storage
- Check that the credentials file path is correct in the Docker run command
- Ensure the credentials file is valid (not expired)
- Contact Kaylee, Chloe, or Adrian for access issues

### BigQuery Errors

If the script can't connect to BigQuery:
- Verify the project ID, dataset ID, and table ID are correct
- Check that data has been loaded into the `hemodyn_table`
- Ensure your service account has query permissions
- Try running a simple query manually to test access

### CUDA/GPU Errors

If you encounter CUDA errors:
- The model will automatically fall back to CPU if CUDA is unavailable
- For GPU training, ensure Docker has GPU access configured
- Reduce batch size if you encounter out-of-memory errors

### Docker Build Issues

If the build fails:
- Ensure Docker is running
- Check your internet connection (needed to download dependencies)
- Try clearing Docker cache: `docker system prune`

## Modifying the Container

### Change Data Source

To use different tables, modify the BigQuery query in `main.py`:

```python
# In load_data_from_bigquery() function
query = f"""
    SELECT id, feature1, feature2, ...
    FROM `{project_id}.{dataset_id}.clinic_table`  # or lab_table
    ...
"""
```

### Modify Model Architecture

Edit `model.py` to change the model structure:

```python
# Create model with custom config
config = {
    'd_model': 256,  # Increase model dimension
    'nhead': 16,     # More attention heads
    'num_encoder_layers': 6,  # Deeper encoder
    ...
}
model = create_model(config)
```

### Add New Dependencies

1. Add packages to `pyproject.toml`:
   ```toml
   dependencies = [
       "package-name>=version",
   ]
   ```
2. Rebuild the Docker image:
   ```bash
   docker build -t baseline-model -f Dockerfile .
   ```

## Project Structure

```
model_container/
├── Dockerfile              # Docker image definition
├── pyproject.toml          # Python dependencies (PyTorch, BigQuery client, etc.)
├── main.py                # Training and evaluation pipeline
├── model.py               # DoseWise model architecture
├── inference.py           # Inference script for predictions
├── test_model.py          # Model testing suite
├── docker.sh              # Convenience script to build and run
└── README.md              # This file

# Generated after training:
├── dosewise_model.pth      # Trained model weights
├── scaler.pkl              # Feature normalization scaler
└── training_history.pkl    # Training metrics
```

## Quick Start Guide

### 1. Test Model Architecture

Before training, verify the model architecture works:

```bash
# Inside container or with PyTorch installed locally
python test_model.py
```

This runs a comprehensive test suite checking:
- Model instantiation
- Forward pass
- Parameter count
- Custom configurations
- Gradient flow
- Device compatibility

### 2. Train Model

```bash
# Basic training
python main.py --train --num_epochs 10

# Training with custom hyperparameters
python main.py --train \
  --batch_size 64 \
  --num_epochs 20 \
  --learning_rate 0.0005 \
  --d_model 256
```

### 3. Make Predictions

```bash
# Predict for a specific patient from BigQuery
python inference.py --patient_id 513 --prediction_length 50

# Predict from CSV file
python inference.py --input_file patient_data.csv --output_file predictions.csv

# Use GPU for inference
python inference.py --patient_id 513 --use_gpu
```

## Model Hyperparameters

Default configuration:
- **Sequence Length**: 100 time steps
- **Prediction Length**: 50 time steps
- **Model Dimension (d_model)**: 128
- **Attention Heads**: 8
- **Encoder Layers**: 4
- **Decoder Layers**: 4
- **Feedforward Dimension**: 512
- **Dropout**: 0.1
- **Batch Size**: 32
- **Learning Rate**: 0.001

Adjust these via command-line arguments (see `--help` for full list).

## Next Steps

After successfully training the model:
1. **Evaluate counterfactual predictions**: Compare control vs treatment outcomes
2. **Integrate with LLM interface**: Use predictions for patient-specific recommendations
3. **Deploy to cloud**: Set up inference endpoint for real-time predictions
4. **Extend to other vitals**: Predict ECG, CO2, PLETH in addition to ABP
5. **Add uncertainty quantification**: Implement prediction intervals

## Research Context

This model implements counterfactual prediction for personalized medicine, enabling clinicians to:
- Predict patient outcomes with and without drug intervention
- Make data-driven dosing decisions
- Understand individual patient responses to medication

## Support

For questions or issues:
- **Technical issues**: Contact Kaylee, Chloe, or Adrian
- **Model architecture**: Refer to the mock_model_diagram.png in `/img`
- **Data pipeline**: See the main project README and ETL documentation

