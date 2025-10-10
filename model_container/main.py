"""
DoseWise Model Training and Inference
Pulls data from BigQuery and trains/evaluates the counterfactual prediction model
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

from model import create_model


class VitalSignsDataset(Dataset):
    """Dataset for patient vital signs time series"""
    
    def __init__(self, data, sequence_length=100, prediction_length=50):
        """
        Args:
            data: DataFrame with columns [id, ART, ECG_II, PLETH, CO2, PHEN_RATE, time]
            sequence_length: Length of input sequence
            prediction_length: Length of prediction sequence
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.data = data
        
        # Group by patient ID
        self.patient_groups = data.groupby('id')
        self.patient_ids = list(self.patient_groups.groups.keys())
        
        # Create sequences for each patient
        self.sequences = []
        self._create_sequences()
    
    def _create_sequences(self):
        """Create overlapping sequences from patient data"""
        for patient_id in self.patient_ids:
            patient_data = self.patient_groups.get_group(patient_id)
            
            # Sort by time
            patient_data = patient_data.sort_values('time')
            
            # Extract features: ART, ECG_II, PLETH, CO2, PHEN_RATE
            features = patient_data[['ART', 'ECG_II', 'PLETH', 'CO2', 'PHEN_RATE']].values
            
            # Create sliding windows
            total_length = self.sequence_length + self.prediction_length
            for i in range(len(features) - total_length + 1):
                input_seq = features[i:i + self.sequence_length]
                target_seq = features[i + self.sequence_length:i + total_length]
                
                # Store only if no NaN values
                if not (np.isnan(input_seq).any() or np.isnan(target_seq).any()):
                    self.sequences.append({
                        'input': input_seq,
                        'target': target_seq[:, 0],  # Predict ART (arterial pressure)
                        'patient_id': patient_id
                    })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            'input': torch.FloatTensor(sequence['input']),
            'target': torch.FloatTensor(sequence['target']),
            'patient_id': sequence['patient_id']
        }


def load_data_from_bigquery(project_id="dosewise-473716", dataset_id="dosewisedb", table_id="hemodyn_table", location="us-central1"):
    """
    Load data from BigQuery
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
    Returns:
        df: Pandas DataFrame with patient data
    """
    print(f"Loading data from BigQuery: {project_id}.{dataset_id}.{table_id} (location: {location})")
    
    client = bigquery.Client(project=project_id, location=location)
    
    # Query to get data
    query = f"""
        SELECT id, ART, ECG_II, PLETH, CO2, PHEN_RATE, time
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE ART IS NOT NULL 
          AND ECG_II IS NOT NULL 
          AND PLETH IS NOT NULL 
          AND CO2 IS NOT NULL 
          AND PHEN_RATE IS NOT NULL
        ORDER BY id, time
    """
    
    print("Executing query...")
    df = client.query(query).to_dataframe()
    
    print(f"Loaded {len(df)} rows from BigQuery")
    print(f"Number of unique patients: {df['id'].nunique()}")
    print(f"\nData shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df


def normalize_data(df, scaler=None, fit=True):
    """
    Normalize vital signs data
    
    Args:
        df: DataFrame with vital signs
        scaler: Pre-fitted scaler (optional)
        fit: Whether to fit the scaler
    Returns:
        df_normalized: Normalized DataFrame
        scaler: Fitted scaler
    """
    feature_columns = ['ART', 'ECG_II', 'PLETH', 'CO2', 'PHEN_RATE']
    
    if scaler is None:
        scaler = StandardScaler()
    
    df_normalized = df.copy()
    
    if fit:
        df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
    else:
        df_normalized[feature_columns] = scaler.transform(df[feature_columns])
    
    return df_normalized, scaler


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the DoseWise model
    
    Args:
        model: DoseWiseModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"\nTraining on {device}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            control_pred, treatment_pred = model(inputs, target_len=targets.size(1))
            
            # For now, use treatment predictions (with drug)
            # Extract first dimension (ART prediction)
            predictions = treatment_pred[:, :, 0]
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                control_pred, treatment_pred = model(inputs, target_len=targets.size(1))
                predictions = treatment_pred[:, :, 0]
                
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model, history


def main(args):
    """Main training and evaluation pipeline"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data from BigQuery
    df = load_data_from_bigquery(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        table_id=args.table_id
    )
    
    # Split by patient ID to avoid data leakage
    patient_ids = df['id'].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)
    
    train_df = df[df['id'].isin(train_ids)]
    val_df = df[df['id'].isin(val_ids)]
    test_df = df[df['id'].isin(test_ids)]
    
    print(f"\nTrain patients: {len(train_ids)}")
    print(f"Val patients: {len(val_ids)}")
    print(f"Test patients: {len(test_ids)}")
    
    # Normalize data
    print("\nNormalizing data...")
    train_df_norm, scaler = normalize_data(train_df, fit=True)
    val_df_norm, _ = normalize_data(val_df, scaler=scaler, fit=False)
    test_df_norm, _ = normalize_data(test_df, scaler=scaler, fit=False)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = VitalSignsDataset(
        train_df_norm,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length
    )
    val_dataset = VitalSignsDataset(
        val_df_norm,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length
    )
    test_dataset = VitalSignsDataset(
        test_df_norm,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length
    )
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nCreating model...")
    model_config = {
        'input_features': 5,
        'output_features': 4,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
    }
    model = create_model(model_config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Train model
    if args.train:
        print("\nStarting training...")
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device
        )
        
        # Save model
        torch.save(model.state_dict(), 'dosewise_model.pth')
        print("\nModel saved to dosewise_model.pth")
        
        # Save training history
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print("Training history saved to training_history.pkl")
    
    # Evaluate on test set
    if args.evaluate:
        print("\nEvaluating model on test set...")
        
        if os.path.exists('dosewise_model.pth'):
            model.load_state_dict(torch.load('dosewise_model.pth'))
            print("Loaded model from dosewise_model.pth")
        
        model = model.to(device)
        model.eval()
        
        test_loss = 0.0
        test_batches = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                control_pred, treatment_pred = model(inputs, target_len=targets.size(1))
                predictions = treatment_pred[:, :, 0]
                
                loss = criterion(predictions, targets)
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        print(f"\nTest Loss (MSE): {avg_test_loss:.4f}")
        print(f"Test RMSE: {np.sqrt(avg_test_loss):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoseWise Model Training")
    
    # Data arguments
    parser.add_argument('--project_id', type=str, default='dosewise-473716',
                        help='GCP project ID')
    parser.add_argument('--dataset_id', type=str, default='dosewisedb',
                        help='BigQuery dataset ID')
    parser.add_argument('--table_id', type=str, default='hemodyn_table',
                        help='BigQuery table ID')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4,
                        help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='Input sequence length')
    parser.add_argument('--prediction_length', type=int, default=50,
                        help='Prediction sequence length')
    
    args = parser.parse_args()
    
    # If no mode specified, default to both
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True
    
    main(args)
