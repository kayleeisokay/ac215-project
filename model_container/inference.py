"""
Inference script for DoseWise model
Load a trained model and make counterfactual predictions on new data
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from model import create_model
from google.cloud import bigquery


def load_trained_model(model_path, scaler_path, device='cpu'):
    """
    Load trained model and scaler
    
    Args:
        model_path: Path to saved model weights (.pth)
        scaler_path: Path to saved scaler (.pkl)
        device: Device to load model on
    Returns:
        model: Loaded model
        scaler: Loaded scaler
    """
    print(f"Loading model from {model_path}...")
    
    # Create model with default config
    # In production, you might want to save the config along with the model
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✓ Model and scaler loaded successfully")
    return model, scaler


def prepare_input_sequence(data, scaler, sequence_length=100):
    """
    Prepare input sequence from patient data
    
    Args:
        data: DataFrame or array with features [ART, ECG_II, PLETH, CO2, PHEN_RATE]
        scaler: Fitted StandardScaler
        sequence_length: Length of input sequence
    Returns:
        input_tensor: Prepared input tensor [1, seq_len, 5]
    """
    # Ensure data is DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(
            data,
            columns=['ART', 'ECG_II', 'PLETH', 'CO2', 'PHEN_RATE']
        )
    
    # Take the last sequence_length time steps
    if len(data) > sequence_length:
        data = data.iloc[-sequence_length:]
    elif len(data) < sequence_length:
        # Pad with first value if too short
        padding_length = sequence_length - len(data)
        padding = pd.DataFrame([data.iloc[0]] * padding_length)
        data = pd.concat([padding, data], ignore_index=True)
    
    # Normalize
    features = data[['ART', 'ECG_II', 'PLETH', 'CO2', 'PHEN_RATE']].values
    features_normalized = scaler.transform(features)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(features_normalized).unsqueeze(0)
    
    return input_tensor


def predict_counterfactual(model, input_tensor, prediction_length=50, device='cpu'):
    """
    Make counterfactual predictions
    
    Args:
        model: Trained DoseWise model
        input_tensor: Input sequence [batch, seq_len, features]
        prediction_length: Number of future time steps to predict
        device: Device to run inference on
    Returns:
        control_predictions: Predictions without drug (u=0)
        treatment_predictions: Predictions with drug (u=dose)
    """
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        control_pred, treatment_pred = model(input_tensor, target_len=prediction_length)
    
    # Extract ABP predictions (first output dimension)
    control_abp = control_pred[:, :, 0].cpu().numpy()
    treatment_abp = treatment_pred[:, :, 0].cpu().numpy()
    
    return control_abp, treatment_abp


def denormalize_predictions(predictions, scaler):
    """
    Denormalize predictions back to original scale
    
    Args:
        predictions: Normalized predictions array
        scaler: Fitted StandardScaler
    Returns:
        denormalized: Predictions in original scale
    """
    # Create dummy array with all features
    n_samples = predictions.shape[1] if predictions.ndim > 1 else predictions.shape[0]
    dummy = np.zeros((n_samples, 5))  # 5 features
    dummy[:, 0] = predictions.squeeze()  # ABP is first feature
    
    # Inverse transform
    denormalized_dummy = scaler.inverse_transform(dummy)
    denormalized = denormalized_dummy[:, 0]
    
    return denormalized


def load_patient_data(patient_id, project_id="dosewise-473716", dataset_id="dosewise", table_id="hemodyn_table"):
    """
    Load specific patient data from BigQuery
    
    Args:
        patient_id: Patient ID to load
        project_id: GCP project ID
        dataset_id: BigQuery dataset
        table_id: BigQuery table
    Returns:
        df: Patient data DataFrame
    """
    print(f"Loading data for patient {patient_id} from BigQuery...")
    
    client = bigquery.Client(project=project_id)
    
    query = f"""
        SELECT ART, ECG_II, PLETH, CO2, PHEN_RATE, time
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE id = {patient_id}
          AND ART IS NOT NULL 
          AND ECG_II IS NOT NULL 
          AND PLETH IS NOT NULL 
          AND CO2 IS NOT NULL 
          AND PHEN_RATE IS NOT NULL
        ORDER BY time
        LIMIT 200
    """
    
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} time steps for patient {patient_id}")
    
    return df


def visualize_predictions(control_pred, treatment_pred, historical_data=None):
    """
    Print and optionally visualize predictions
    
    Args:
        control_pred: Control predictions (no drug)
        treatment_pred: Treatment predictions (with drug)
        historical_data: Optional historical ABP values
    """
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nControl Branch (No Drug) - ABP Predictions:")
    print(f"  Mean: {control_pred.mean():.2f}")
    print(f"  Std:  {control_pred.std():.2f}")
    print(f"  Min:  {control_pred.min():.2f}")
    print(f"  Max:  {control_pred.max():.2f}")
    
    print(f"\nTreatment Branch (With Drug) - ABP Predictions:")
    print(f"  Mean: {treatment_pred.mean():.2f}")
    print(f"  Std:  {treatment_pred.std():.2f}")
    print(f"  Min:  {treatment_pred.min():.2f}")
    print(f"  Max:  {treatment_pred.max():.2f}")
    
    # Calculate treatment effect
    effect = treatment_pred.mean() - control_pred.mean()
    effect_pct = (effect / control_pred.mean()) * 100
    
    print(f"\nEstimated Treatment Effect:")
    print(f"  Absolute: {effect:+.2f} (mmHg)")
    print(f"  Relative: {effect_pct:+.2f}%")
    
    if historical_data is not None:
        print(f"\nHistorical ABP (last 10 time steps):")
        print(f"  {historical_data[-10:]}")
    
    print("\n" + "="*60)


def main(args):
    """Main inference pipeline"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load trained model and scaler
    model, scaler = load_trained_model(
        args.model_path,
        args.scaler_path,
        device=device
    )
    
    # Load patient data
    if args.patient_id:
        # Load from BigQuery
        patient_data = load_patient_data(
            args.patient_id,
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            table_id=args.table_id
        )
    elif args.input_file:
        # Load from CSV file
        print(f"Loading data from {args.input_file}...")
        patient_data = pd.read_csv(args.input_file)
    else:
        raise ValueError("Must provide either --patient_id or --input_file")
    
    # Store historical ABP for comparison
    historical_abp = patient_data['ART'].values
    
    # Prepare input sequence
    input_tensor = prepare_input_sequence(
        patient_data,
        scaler,
        sequence_length=args.sequence_length
    )
    
    print(f"Input sequence shape: {input_tensor.shape}")
    
    # Make predictions
    print(f"\nPredicting {args.prediction_length} future time steps...")
    control_pred, treatment_pred = predict_counterfactual(
        model,
        input_tensor,
        prediction_length=args.prediction_length,
        device=device
    )
    
    # Denormalize predictions
    control_pred_denorm = denormalize_predictions(control_pred, scaler)
    treatment_pred_denorm = denormalize_predictions(treatment_pred, scaler)
    
    # Visualize results
    visualize_predictions(
        control_pred_denorm,
        treatment_pred_denorm,
        historical_data=historical_abp
    )
    
    # Save predictions if requested
    if args.output_file:
        results = pd.DataFrame({
            'time_step': range(len(control_pred_denorm)),
            'control_prediction': control_pred_denorm,
            'treatment_prediction': treatment_pred_denorm,
            'treatment_effect': treatment_pred_denorm - control_pred_denorm
        })
        results.to_csv(args.output_file, index=False)
        print(f"\n✓ Predictions saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoseWise Model Inference")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='dosewise_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl',
                        help='Path to fitted scaler')
    
    # Data arguments
    parser.add_argument('--patient_id', type=int,
                        help='Patient ID to load from BigQuery')
    parser.add_argument('--input_file', type=str,
                        help='CSV file with patient data')
    parser.add_argument('--output_file', type=str,
                        help='Output file to save predictions')
    
    # BigQuery arguments (if using patient_id)
    parser.add_argument('--project_id', type=str, default='dosewise-473716',
                        help='GCP project ID')
    parser.add_argument('--dataset_id', type=str, default='dosewise',
                        help='BigQuery dataset ID')
    parser.add_argument('--table_id', type=str, default='hemodyn_table',
                        help='BigQuery table ID')
    
    # Inference arguments
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='Input sequence length')
    parser.add_argument('--prediction_length', type=int, default=50,
                        help='Prediction sequence length')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')
    
    args = parser.parse_args()
    
    main(args)

