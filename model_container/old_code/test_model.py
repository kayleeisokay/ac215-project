"""
Quick test script to verify model architecture and forward pass
"""

import torch
from model import create_model, DoseWiseModel


def test_model_instantiation():
    """Test that model can be created"""
    print("Testing model instantiation...")
    model = create_model()
    assert isinstance(model, DoseWiseModel)
    print("✓ Model instantiated successfully")
    return model


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    # Create model
    model = create_model()
    model.eval()
    
    # Create dummy input
    batch_size = 4
    seq_len = 100
    input_features = 5  # ABP, PLETH, CO2, ECG, u
    
    dummy_input = torch.randn(batch_size, seq_len, input_features)
    
    # Forward pass
    with torch.no_grad():
        control_pred, treatment_pred = model(dummy_input, target_len=50)
    
    # Check output shapes
    assert control_pred.shape == (batch_size, 50, 4), f"Expected (4, 50, 4), got {control_pred.shape}"
    assert treatment_pred.shape == (batch_size, 50, 4), f"Expected (4, 50, 4), got {treatment_pred.shape}"
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Control predictions shape: {control_pred.shape}")
    print(f"✓ Treatment predictions shape: {treatment_pred.shape}")
    
    return control_pred, treatment_pred


def test_parameter_count():
    """Test and display parameter count"""
    print("\nTesting parameter count...")
    
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    assert total_params > 0, "Model has no parameters"
    assert trainable_params == total_params, "Some parameters are frozen"
    
    return total_params


def test_custom_config():
    """Test model with custom configuration"""
    print("\nTesting custom configuration...")
    
    custom_config = {
        'input_features': 5,
        'output_features': 4,
        'd_model': 64,  # Smaller model
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
    }
    
    model = create_model(custom_config)
    batch_size = 2
    seq_len = 50
    
    dummy_input = torch.randn(batch_size, seq_len, 5)
    
    with torch.no_grad():
        control_pred, treatment_pred = model(dummy_input, target_len=25)
    
    assert control_pred.shape == (batch_size, 25, 4)
    assert treatment_pred.shape == (batch_size, 25, 4)
    
    print("✓ Custom configuration works correctly")
    return model


def test_gradient_flow():
    """Test that gradients flow through the model"""
    print("\nTesting gradient flow...")
    
    model = create_model()
    model.train()
    
    # Create dummy data
    batch_size = 2
    seq_len = 50
    dummy_input = torch.randn(batch_size, seq_len, 5, requires_grad=True)
    dummy_target = torch.randn(batch_size, 25)
    
    # Forward pass
    control_pred, treatment_pred = model(dummy_input, target_len=25)
    
    # Compute loss and backward
    loss = torch.nn.functional.mse_loss(treatment_pred[:, :, 0], dummy_target)
    loss.backward()
    
    # Check that gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients computed"
    
    print(f"✓ Loss computed: {loss.item():.4f}")
    print("✓ Gradients flow correctly through the model")


def test_device_compatibility():
    """Test CPU/CUDA compatibility"""
    print("\nTesting device compatibility...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    model = create_model().to(device)
    dummy_input = torch.randn(2, 50, 5).to(device)
    
    with torch.no_grad():
        control_pred, treatment_pred = model(dummy_input, target_len=25)
    
    assert control_pred.device.type == device
    assert treatment_pred.device.type == device
    
    print(f"✓ Model works correctly on {device}")


def main():
    """Run all tests"""
    print("="*60)
    print("DoseWise Model Test Suite")
    print("="*60)
    
    try:
        # Run tests
        test_model_instantiation()
        test_forward_pass()
        test_parameter_count()
        test_custom_config()
        test_gradient_flow()
        test_device_compatibility()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ TEST FAILED: {str(e)}")
        print("="*60)
        raise


if __name__ == "__main__":
    main()

