"""
Demo inference script using the trained demo model
"""

import torch
import numpy as np
from model import create_model


def demo_inference():
    """Run inference demo with synthetic patient data"""
    
    print("="*60)
    print("DoseWise Inference Demo")
    print("="*60)
    
    # Load trained model
    print("\n✓ Loading trained model...")
    model = create_model()
    model.load_state_dict(torch.load('demo_model.pth'))
    model.eval()
    print("  Model loaded successfully")
    
    # Create synthetic patient data
    print("\n✓ Creating synthetic patient vital signs...")
    seq_len = 100
    t = np.linspace(0, 10, seq_len)
    
    # Simulate vital signs
    art = 120 + 20 * np.sin(t) + np.random.randn(seq_len) * 5
    ecg = 0.8 + 0.3 * np.sin(2 * t) + np.random.randn(seq_len) * 0.1
    pleth = 30 + 10 * np.sin(1.5 * t) + np.random.randn(seq_len) * 3
    co2 = np.random.randn(seq_len) * 0.5
    phen = np.random.rand(seq_len) * 2  # Drug dosage
    
    # Stack features
    patient_data = np.stack([art, ecg, pleth, co2, phen], axis=1)
    
    print(f"  Input sequence shape: {patient_data.shape}")
    print(f"  Last 5 ABP values: {art[-5:]}")
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(patient_data).unsqueeze(0)
    
    # Make predictions
    print("\n✓ Generating counterfactual predictions...")
    prediction_length = 50
    
    with torch.no_grad():
        control_pred, treatment_pred = model(input_tensor, target_len=prediction_length)
    
    # Extract ABP predictions
    control_abp = control_pred[0, :, 0].numpy()
    treatment_abp = treatment_pred[0, :, 0].numpy()
    
    print(f"\n  Prediction Results for next {prediction_length} time steps:")
    print("  " + "-"*56)
    
    # Show predictions
    print(f"\n  Control Branch (No Drug):")
    print(f"    Mean ABP: {control_abp.mean():.2f}")
    print(f"    Std ABP:  {control_abp.std():.2f}")
    print(f"    Range:    {control_abp.min():.2f} to {control_abp.max():.2f}")
    print(f"    First 5:  {control_abp[:5]}")
    
    print(f"\n  Treatment Branch (With Drug):")
    print(f"    Mean ABP: {treatment_abp.mean():.2f}")
    print(f"    Std ABP:  {treatment_abp.std():.2f}")
    print(f"    Range:    {treatment_abp.min():.2f} to {treatment_abp.max():.2f}")
    print(f"    First 5:  {treatment_abp[:5]}")
    
    # Calculate treatment effect
    effect = treatment_abp - control_abp
    avg_effect = effect.mean()
    
    print(f"\n  Counterfactual Treatment Effect:")
    print(f"    Mean difference: {avg_effect:+.2f}")
    print(f"    Effect range: {effect.min():+.2f} to {effect.max():+.2f}")
    
    if avg_effect > 0:
        print(f"    → Drug increases ABP by ~{abs(avg_effect):.2f} on average")
    else:
        print(f"    → Drug decreases ABP by ~{abs(avg_effect):.2f} on average")
    
    print("\n" + "="*60)
    print("✓ Inference Demo Complete!")
    print("="*60)
    
    return control_abp, treatment_abp


if __name__ == "__main__":
    control, treatment = demo_inference()

