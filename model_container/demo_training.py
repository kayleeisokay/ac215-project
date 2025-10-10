"""
Quick demo of model training with synthetic data
This shows how the training works without needing BigQuery access
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from model import create_model


class SyntheticVitalSignsDataset(Dataset):
    """Generate synthetic patient data for testing"""
    
    def __init__(self, num_samples=1000, sequence_length=100, prediction_length=50):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Generate synthetic data
        self.data = []
        for i in range(num_samples):
            # Create realistic-looking time series
            t = np.linspace(0, 10, sequence_length)
            
            # Simulate vital signs with some patterns
            art = 120 + 20 * np.sin(t) + np.random.randn(sequence_length) * 5
            ecg = 0.8 + 0.3 * np.sin(2 * t) + np.random.randn(sequence_length) * 0.1
            pleth = 30 + 10 * np.sin(1.5 * t) + np.random.randn(sequence_length) * 3
            co2 = np.random.randn(sequence_length) * 0.5
            phen = np.random.rand(sequence_length) * 2
            
            input_seq = np.stack([art, ecg, pleth, co2, phen], axis=1)
            
            # Target is future ART values (with some trend)
            target_t = np.linspace(10, 15, prediction_length)
            target = 120 + 20 * np.sin(target_t) + np.random.randn(prediction_length) * 5
            
            self.data.append({
                'input': input_seq,
                'target': target
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input': torch.FloatTensor(self.data[idx]['input']),
            'target': torch.FloatTensor(self.data[idx]['target'])
        }


def train_demo(num_epochs=5, batch_size=16):
    """Run a quick training demo"""
    
    print("="*60)
    print("DoseWise Model Training Demo")
    print("="*60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")
    
    # Create synthetic datasets
    print("\n✓ Creating synthetic datasets...")
    train_dataset = SyntheticVitalSignsDataset(num_samples=800)
    val_dataset = SyntheticVitalSignsDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\n✓ Creating model...")
    model = create_model()
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  - Model parameters: {num_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n✓ Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            control_pred, treatment_pred = model(inputs, target_len=targets.size(1))
            predictions = treatment_pred[:, :, 0]
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
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
        
        print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Test predictions
    print("\n✓ Testing predictions...")
    model.eval()
    
    # Get a sample
    sample = val_dataset[0]
    input_seq = sample['input'].unsqueeze(0).to(device)
    target_seq = sample['target'].numpy()
    
    with torch.no_grad():
        control_pred, treatment_pred = model(input_seq, target_len=50)
    
    control_values = control_pred[0, :, 0].cpu().numpy()
    treatment_values = treatment_pred[0, :, 0].cpu().numpy()
    
    print(f"\n  Sample Prediction Results:")
    print(f"  - True ABP (first 5 steps): {target_seq[:5]}")
    print(f"  - Control Pred (first 5):   {control_values[:5]}")
    print(f"  - Treatment Pred (first 5): {treatment_values[:5]}")
    
    # Calculate treatment effect
    effect = treatment_values.mean() - control_values.mean()
    print(f"\n  Estimated Treatment Effect: {effect:+.2f}")
    
    # Save model
    print("\n✓ Saving model...")
    torch.save(model.state_dict(), 'demo_model.pth')
    print("  - Saved to: demo_model.pth")
    
    print("\n" + "="*60)
    print("✓ Demo Training Complete!")
    print("="*60)
    
    return model


if __name__ == "__main__":
    model = train_demo(num_epochs=5, batch_size=16)

