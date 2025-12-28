# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
import torch
import numpy as np
import sys
import os


from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

# Import your specific FMTK components
from fmtk.pipeline import Pipeline
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.decoders.classification.svm import SVMDecoder
from fmtk.components.decoders.classification.mlp import MLPDecoder
from fmtk.logger import Logger

# --- 1. Load the Model (Runs once at startup) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading FMTK model on {device}...")

# Initialize Logger
# logger = Logger(device, 'triton_server')

# Initialize Backbone (Chronos) & Pipeline
backbone = ChronosModel(device, 'tiny')
pipeline = Pipeline(backbone)

# Initialize Decoder (SVM)
decoder = MLPDecoder(device,{'input_dim':256,'output_dim':5,'hidden_dim':128})
# NOTE: In a real app, load your trained weights here:
# decoder.model.load_state_dict(torch.load("path/to/weights.pth"))
pipeline.add_decoder(decoder, load=True, train=False, path="ecgclass_chronostiny_mlp")
pipeline.set_eval_mode()

print("Model loaded successfully!")

# --- 2. Define the Inference Function ---
@batch
def infer_fn(input_signal):
    """
    Receives a batch of inputs from Triton.
    input_signal: NumPy array of shape (Batch, Time, Channel) or (Batch, Time)
    """
    # Convert NumPy -> PyTorch
    input_tensor = torch.from_numpy(input_signal)
    
    # Run Inference
    with torch.no_grad():
        # FMTK forward pass
        logits = pipeline.forward(x=input_tensor)
        # Convert PyTorch -> NumPy (Triton expects NumPy)
        output_np = logits.cpu().numpy()

    return {"output_prediction": output_np}

# --- 3. Start the Server ---
if __name__ == "__main__":
    # Define inputs/outputs. 
    # Shape (-1, 1) means [Any_Length, 1_Channel]
    config = TritonConfig(http_port=8001,grpc_port=8050)
    with Triton(config=config) as p:
        p.bind(
            model_name="fmtk_chronos",
            infer_func=infer_fn,
            inputs=[
                Tensor(name="input_signal", dtype=np.float32, shape=(-1, 1)),
            ],
            outputs=[
                Tensor(name="output_prediction", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=16)
        )
        
        print("\n✅ Triton Server started! Listening")
        print("Press Ctrl+C to stop.\n")
        p.serve()