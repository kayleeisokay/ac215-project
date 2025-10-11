import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from sklearn.preprocessing import StandardScaler

import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"dosewise-473716-9f4874e812d6.json"

import os


def debug_credentials():
    creds_path = "/credentials"
    if os.path.exists(creds_path):
        print(f"Contents of /credentials:")
        for item in os.listdir(creds_path):
            full_path = os.path.join(creds_path, item)
            if os.path.isdir(full_path):
                print(f"  📁 {item}/ (directory)")
            else:
                print(f"  📄 {item} (file)")

        # Check the specific file
        json_path = "/credentials/dosewise-473716-9f4874e812d6.json"
        if os.path.exists(json_path):
            if os.path.isdir(json_path):
                print(f"❌ {json_path} is a DIRECTORY, not a file!")
            else:
                print(f"✅ {json_path} is a file and exists!")
        else:
            print(f"❌ {json_path} does not exist!")
    else:
        print("❌ /credentials directory does not exist!")


class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use last time step
        return out


class FlexibleLSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(FlexibleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Single LSTM that can handle both cases
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, use_phen=True):
        if not use_phen:
            # Mask out PHEN_RATE by setting it to zero
            x = x.clone()
            x[:, :, 4] = 0  # Set PHEN_RATE (index 4) to zero

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MedicalPredictor:
    def __init__(self):
        self.model = FlexibleLSTMModel(
            input_size=5, hidden_size=50, num_layers=2, output_size=1
        )
        self.scaler = StandardScaler()
        self.sequence_length = 10  # Use last 10 time steps to predict next ART

    def save_model(self, base_path="medical_model"):
        """Save model, scaler, and metadata locally with timestamp versioning"""
        import os
        from datetime import datetime

        # Create base directory
        os.makedirs(base_path, exist_ok=True)

        # Add timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.pth"
        local_path = f"{base_path}/{model_filename}"

        # Save model state with timestamp
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "input_size": 5,
                    "hidden_size": 50,
                    "num_layers": 2,
                    "output_size": 1,
                },
                "scaler_mean": (
                    self.scaler.mean_ if hasattr(self.scaler, "mean_") else None
                ),
                "scaler_scale": (
                    self.scaler.scale_ if hasattr(self.scaler, "scale_") else None
                ),
                "sequence_length": self.sequence_length,
                "timestamp": timestamp,  # Include timestamp in model metadata
            },
            local_path,
        )

        print(f"Model saved to {local_path}")
        return local_path  # Return full path for bucket upload

    def save_to_bucket(self, bucket_name, local_path):
        """Upload model files to Google Cloud Storage with timestamp versioning"""
        import os

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Extract filename from local_path
        filename = os.path.basename(local_path)

        # Upload model file with timestamped name
        model_blob = bucket.blob(f"models/{filename}")
        model_blob.upload_from_filename(local_path)

        print(f"Model uploaded to gs://{bucket_name}/models/{filename}")

    def load_data_from_bigquery(self):
        """Fetch medical time series data from BigQuery - sampled for speed"""
        client = bigquery.Client()

        query = """
        WITH top_patients AS (
        SELECT DISTINCT id
        FROM `dosewise-473716.dosewisedb.hemodyn_table`
        ORDER BY id
        LIMIT 5
        )
        SELECT 
            h.id, h.ART, h.ECG_II, h.PLETH, h.CO2, h.PHEN_RATE, h.time
        FROM `dosewise-473716.dosewisedb.hemodyn_table` h
        INNER JOIN top_patients t ON h.id = t.id
        ORDER BY h.id, h.time
        """

        results = client.query(query).result()
        df = pd.DataFrame([dict(row) for row in results])

        # Debug info
        print(f"Loaded {len(df)} rows from {df['id'].nunique()} patients")
        print(f"Patient IDs: {sorted(df['id'].unique())}")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        print(
            f"Features summary:\n{df[['ART', 'ECG_II', 'PLETH', 'CO2', 'PHEN_RATE']].describe()}"
        )

        return df

    def prepare_sequences(self, df):
        """Convert time series data to sequences with per-patient scaling"""
        features = ["ART", "ECG_II", "PLETH", "CO2", "PHEN_RATE"]

        all_sequences_X = []
        all_sequences_y = []

        for patient_id, patient_data in df.groupby("id"):
            patient_data = patient_data.sort_values("time")
            data = patient_data[features].dropna().values

            if len(data) > self.sequence_length:
                # Scale PER PATIENT (crucial!)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)

                X_patient, y_patient = [], []
                for i in range(len(scaled_data) - self.sequence_length):
                    X_patient.append(scaled_data[i : (i + self.sequence_length)])
                    y_patient.append(scaled_data[i + self.sequence_length, 0])

                all_sequences_X.extend(X_patient)
                all_sequences_y.extend(y_patient)

        return np.array(all_sequences_X), np.array(all_sequences_y)

    def train_model(self, X, y, epochs=50):
        """Train the flexible LSTM model with both configurations"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Get predictions with PHEN_RATE
            pred_with_phen = self.model(X_tensor, use_phen=True)
            loss_with_phen = criterion(pred_with_phen, y_tensor)

            # Get predictions without PHEN_RATE
            pred_without_phen = self.model(X_tensor, use_phen=False)
            loss_without_phen = criterion(pred_without_phen, y_tensor)

            # Total loss (both branches)
            total_loss = loss_with_phen + loss_without_phen

            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], "
                    f"With PHEN: {loss_with_phen.item():.4f}, "
                    f"Without PHEN: {loss_without_phen.item():.4f}"
                )

    def predict_sequence(self, input_sequence, num_predictions=600):
        """Generate a sequence of predictions for the next 10 minutes"""
        self.model.eval()
        with torch.no_grad():
            # Start with the initial input sequence
            current_sequence = torch.FloatTensor(input_sequence).unsqueeze(
                0
            )  # shape: (1, seq_len, 5)
            predictions_with_phen = []
            predictions_without_phen = []
            phen_effects = []

            for i in range(num_predictions):
                # Get predictions for current sequence
                pred_with_phen = self.model(current_sequence, use_phen=True)
                pred_without_phen = self.model(current_sequence, use_phen=False)

                # Store predictions
                pred_with_phen_val = pred_with_phen.item()
                pred_without_phen_val = pred_without_phen.item()
                phen_effect_val = pred_with_phen_val - pred_without_phen_val

                predictions_with_phen.append(pred_with_phen_val)
                predictions_without_phen.append(pred_without_phen_val)
                phen_effects.append(phen_effect_val)

                # Update the sequence for next prediction (autoregressive)
                # Remove oldest time step, add new prediction
                if i < num_predictions - 1:
                    # Create new sequence by shifting window
                    new_sequence = current_sequence[
                        :, 1:, :
                    ].clone()  # Remove oldest time step

                    # Create new time step with predicted ART and current features
                    last_timestep = current_sequence[
                        :, -1:, :
                    ].clone()  # Get last timestep

                    # Update ART with prediction (with phen version)
                    last_timestep[:, :, 0] = pred_with_phen  # ART is index 0

                    # Append new timestep
                    current_sequence = torch.cat([new_sequence, last_timestep], dim=1)

                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{num_predictions} predictions...")

            return {
                "predictions_with_phen": predictions_with_phen,
                "predictions_without_phen": predictions_without_phen,
                "phen_effects": phen_effects,
                "timestamps": list(range(1, num_predictions + 1)),  # 1 to 600 seconds
            }

    def run(self):
        """Main pipeline"""
        print("Loading data from BigQuery...")
        df = self.load_data_from_bigquery()

        print("Preparing sequences...")
        X, y = self.prepare_sequences(df)

        print(f"Training flexible model on {len(X)} sequences...")
        self.train_model(X, y, epochs=10)

        print("Model training completed!")

        # Generate 10 minutes of predictions (600 seconds)
        if len(X) > 0:
            print("Generating 10-minute prediction sequence (600 predictions)...")
            predictions = self.predict_sequence(X[0], num_predictions=600)

            # Print summary statistics
            print(f"\n=== 10-MINUTE PREDICTION SUMMARY ===")
            print(
                f"With PHEN_RATE - Mean: {np.mean(predictions['predictions_with_phen']):.4f}, "
                f"Std: {np.std(predictions['predictions_with_phen']):.4f}"
            )
            print(
                f"Without PHEN_RATE - Mean: {np.mean(predictions['predictions_without_phen']):.4f}, "
                f"Std: {np.std(predictions['predictions_without_phen']):.4f}"
            )
            print(
                f"PHEN_RATE effect - Mean: {np.mean(predictions['phen_effects']):.4f}, "
                f"Std: {np.std(predictions['phen_effects']):.4f}"
            )

            # Print first 10 predictions as sample
            print(f"\nFirst 10 predictions (seconds 1-10):")
            for i in range(10):
                print(
                    f"Sec {i+1}: With_PHEN={predictions['predictions_with_phen'][i]:.4f}, "
                    f"Without_PHEN={predictions['predictions_without_phen'][i]:.4f}, "
                    f"Effect={predictions['phen_effects'][i]:.4f}"
                )


if __name__ == "__main__":
    debug_credentials()

    predictor = MedicalPredictor()
    predictor.run()

    predictor.save_model("trained_model")

    # Upload to Google Cloud Storage
    model_path = predictor.save_model("trained_model")

    # Pass the FILE path, not the directory
    predictor.save_to_bucket("dosewisedb", model_path)
