import torch
import numpy as np
import sys
import os
import json

# Triton Python Backend Utils
import triton_python_backend_utils as pb_utils

# --- DEPENDENCY SETUP ---
# Ensure python can find your 'fmtk' package. 
# If 'fmtk' is installed globally in the container, you can remove this.
# Otherwise, we add the model directory to path.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import your specific FMTK components
from fmtk.pipeline import Pipeline
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.decoders.classification.mlp import MLPDecoder

class TritonPythonModel:
    def initialize(self, args):
        """
        Called once when the model is loaded.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[fmtk_chronos] Loading model on {self.device}...", flush=True)

        # 1. Initialize Backbone (Chronos) & Pipeline
        # We need to ensure the config allows the model to find files relative to this dir
        self.backbone = ChronosModel(self.device, 'tiny')
        self.pipeline = Pipeline(self.backbone)

        # 2. Initialize Decoder
        decoder = MLPDecoder(self.device, {'input_dim': 256, 'output_dim': 5, 'hidden_dim': 128})
        
        # 3. Load Weights
        # Assuming weights are located in the same folder as model.py
        weights_path = os.path.join(args['model_repository'], args['model_version'], "ecgclass_chronostiny_mlp")
        
        # Note: Ensure your pipeline.add_decoder handles absolute paths correctly
        self.pipeline.add_decoder(decoder, load=True, train=False, path=weights_path)
        self.pipeline.set_eval_mode()
        
        print("[fmtk_chronos] Model loaded successfully!", flush=True)

    def execute(self, requests):
        """
        Receives a list of requests (batch).
        """
        responses = []
        
        # Collect inputs from all requests to form a batch
        # Note: This simple stacking assumes all inputs in the batch have the SAME sequence length.
        # If requests have different lengths, you must pad them here or handle them individually.
        input_list = []
        
        for request in requests:
            # Get the input tensor by name (must match config.pbtxt)
            in_0 = pb_utils.get_input_tensor_by_name(request, "input_signal")
            
            # Convert to numpy, then torch
            # Clone is sometimes needed if memory layout is non-contiguous
            tensor = torch.as_tensor(in_0.as_numpy(), device=self.device)
            input_list.append(tensor)

        # Create a batch (Stacking)
        # Shape becomes: [Batch_Size, Time, Channels]
        # If inputs vary in Time length, this stack will fail without padding logic.
        try:
            batched_input = torch.stack(input_list, dim=0)
        except Exception as e:
            # Fallback for debugging: If stacking fails, you might need to process individually
            # or pad. For now, we return error.
             return [pb_utils.InferenceResponse(error=pb_utils.TritonError(f"Batching failed: {e}")) for _ in requests]

        # --- INFERENCE ---
        with torch.no_grad():
            # FMTK forward pass
            logits = self.pipeline.forward(x=batched_input)
            
            # Output is expected to be [Batch, Output_Dim]
            output_np = logits.cpu().numpy()

        # --- PACK RESPONSES ---
        # We must split the output batch back into individual responses
        for i, request in enumerate(requests):
            # Slice the output for this specific request
            single_output = output_np[i] # Shape (Output_Dim,)
            
            # Create output tensor
            out_tensor = pb_utils.Tensor("output_prediction", single_output)
            
            # Create response
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        """
        Clean up resources.
        """
        print("[fmtk_chronos] Cleaning up...", flush=True)