#!/usr/bin/env python3
import torch
import numpy as np
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# Request model
class PredictionRequest(BaseModel):
    model_path: str  # gs://dosewisedb/models/model_20251011_023045.pth
    input_sequence: list  # 10 timesteps × 5 features
    use_phen: bool = True


class PredictionResponse(BaseModel):
    prediction: float
    model_used: str
    timestamp: str


# Model definition (must match training)
class FlexibleLSTMModel(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, use_phen=True):
        if not use_phen:
            x = x.clone()
            x[:, :, 4] = 0
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ModelInference:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.loaded_model_path = None

    def load_model_from_gcs(self, gcs_path):
        """Load model from GCS bucket"""
        if self.loaded_model_path == gcs_path:
            return  # Already loaded

        client = storage.Client()

        # Parse GCS path
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path[5:]
        bucket_name, blob_path = gcs_path.split("/", 1)

        # Download model
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as temp_file:
            blob.download_to_filename(temp_file.name)
            checkpoint = torch.load(temp_file.name, map_location="cpu")

        # Load model
        self.model = FlexibleLSTMModel(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # FIX: Properly recreate scaler with all required attributes
        self.scaler = StandardScaler()
        if (
            checkpoint.get("scaler_mean") is not None
            and checkpoint.get("scaler_scale") is not None
        ):

            # Reconstruct scaler state
            self.scaler.mean_ = np.array(checkpoint["scaler_mean"])
            self.scaler.scale_ = np.array(checkpoint["scaler_scale"])
            self.scaler.var_ = np.array(
                checkpoint.get("scaler_var", checkpoint["scaler_scale"] ** 2)
            )
            self.scaler.n_samples_seen_ = checkpoint.get("n_samples_seen", 1)

            print(f"✅ Scaler loaded with mean: {self.scaler.mean_[:3]}...")
        else:
            print("⚠️  No scaler state found in checkpoint")
            # Create a dummy fitted scaler to avoid errors
            self.scaler.mean_ = np.zeros(5)
            self.scaler.scale_ = np.ones(5)
            self.scaler.var_ = np.ones(5)
            self.scaler.n_samples_seen_ = 1

        self.loaded_model_path = gcs_path
        print(f"✅ Model loaded: {gcs_path}")

    def predict(self, input_sequence, use_phen=True):
        """Make prediction"""
        if self.model is None:
            raise ValueError("No model loaded")

        # Convert to numpy
        input_array = np.array(input_sequence)

        # FIX: Check if scaler is fitted
        if not hasattr(self.scaler, "mean_") or self.scaler.mean_ is None:
            print("⚠️  Scaler not fitted, using raw input")
            scaled_input = input_array  # Use raw data
        else:
            scaled_input = self.scaler.transform(input_array)

        # Convert to tensor and predict
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(input_tensor, use_phen=use_phen)

        return prediction.item()


# Initialize FastAPI app
app = FastAPI(title="Medical Model Inference")
inference_engine = ModelInference()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Load model if needed
        inference_engine.load_model_from_gcs(request.model_path)

        # Make prediction
        prediction = inference_engine.predict(request.input_sequence, request.use_phen)

        return PredictionResponse(
            prediction=prediction,
            model_used=request.model_path,
            timestamp=np.datetime64("now").astype(str),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add this endpoint to your existing inference.py
class SequencePredictionRequest(BaseModel):
    model_path: str
    input_sequence: list  # 10 timesteps × 5 features
    num_predictions: int = 600
    use_phen: bool = True


@app.post("/predict_sequence")
async def predict_sequence(request: SequencePredictionRequest):
    try:
        inference_engine.load_model_from_gcs(request.model_path)

        # Convert input to numpy
        input_array = np.array(request.input_sequence)

        # Generate sequence predictions (like your training code)
        predictions_with_phen = []
        predictions_without_phen = []
        phen_effects = []

        # Start with initial sequence
        current_sequence = input_array.copy()

        for i in range(request.num_predictions):
            # Scale current sequence
            scaled_input = inference_engine.scaler.transform(current_sequence)
            input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)

            # Get predictions
            with torch.no_grad():
                pred_with_phen = inference_engine.model(input_tensor, use_phen=True)
                pred_without_phen = inference_engine.model(input_tensor, use_phen=False)

            pred_with_val = pred_with_phen.item()
            pred_without_val = pred_without_phen.item()
            effect_val = pred_with_val - pred_without_val

            predictions_with_phen.append(pred_with_val)
            predictions_without_phen.append(pred_without_val)
            phen_effects.append(effect_val)

            # Update sequence for next prediction (autoregressive)
            if i < request.num_predictions - 1:
                # Remove oldest, add new prediction (like your training code)
                new_sequence = current_sequence[1:]  # Remove first timestep

                # Create new timestep with predicted ART and current features
                new_timestep = current_sequence[-1].copy()
                new_timestep[0] = pred_with_val  # Update ART

                current_sequence = np.vstack([new_sequence, new_timestep])

        return {
            "predictions_with_phen": predictions_with_phen,
            "predictions_without_phen": predictions_without_phen,
            "phen_effects": phen_effects,
            "timestamps": list(range(1, request.num_predictions + 1)),
            "model_used": request.model_path,
            "num_predictions": request.num_predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
