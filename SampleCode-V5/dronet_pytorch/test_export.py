import sys
import os
import torch

def verify_exported_pt(pt_path="dronet_android.pt"):
    """
    Verification script for exported Android TorchScript model (.pt).
    Checks:
    1. Model loads properly via torch.jit.load
    2. Accepts dummy input of shape [1, 3, 224, 224]
    3. Produces output tensor of shape [1, 3] representing [Roll, Pitch, Yaw]
    """
    if not os.path.exists(pt_path):
        print(f"ERROR: Exported model file '{pt_path}' does not exist.")
        sys.exit(1)

    print(f"--- Verifying Exported TorchScript Model: {pt_path} ---")

    # 1. Load TorchScript model
    try:
        model = torch.jit.load(pt_path)
        model.eval()
        print("1. torch.jit.load() successful.")
    except Exception as e:
        print(f"FAILED to load TorchScript model: {e}")
        sys.exit(1)

    # 2. Prepare dummy input: Batch size 1, 3 channels (RGB), 224x224 height/width
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    print(f"2. Input shape prepared: {dummy_input.shape}")

    # 3. Perform inference
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"3. Forward pass successful. Output: {output}")
        except Exception as e:
            print(f"FAILED during forward pass: {e}")
            sys.exit(1)

    # 4. Validate output shape
    expected_shape = torch.Size([1, 3])
    print(f"4. Checking output shape... Output shape is {output.shape}, Expected is {expected_shape}")

    assert output.shape == expected_shape, f"Validation Failed: Expected shape {expected_shape}, but got {output.shape}"

    print("\n=======================================================")
    print(" VERIFICATION PASSED: Model is ready for Android (PyTorch Mobile 1.13.1)!")
    print("=======================================================\n")

if __name__ == "__main__":
    pt_file = sys.argv[1] if len(sys.argv) > 1 else "dronet_android.pt"
    verify_exported_pt(pt_file)
