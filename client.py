import numpy as np
from pytriton.client import ModelClient

# Generate dummy time-series data
input_data = np.random.randn(1, 100, 1).astype(np.float32)

# CHANGE 1: Use the full URL matching your server logs (HTTP usually defaults to 8000)
# If you prefer gRPC, use "grpc://localhost:8001"
url = "http://login2:8001"

try:
    print(f"Connecting to {url}...")
    with ModelClient(url, "fmtk_chronos") as client:
        
        # CHANGE 2: Wait for the specific model to be ready
        print("Waiting for model 'fmtk_chronos' to load...")
        client.wait_for_model(timeout_s=60)
        print("Model is ready! Sending inference request...")

        # Inference
        result_dict = client.infer_batch(input_signal=input_data)
        
        print("Prediction received:")
        print(result_dict["output_prediction"])

except Exception as e:
    print(f"An error occurred: {e}")