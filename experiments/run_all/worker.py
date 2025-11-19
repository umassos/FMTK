# worker.py
import json
import torch
import gc
from inference_pipeline import InferencePipeline
import sys
if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    task_name  = payload["task_name"]
    task_info  = payload["task_info"]
    pipeline   = payload["pipeline"]

    pipe = InferencePipeline(task_name, task_info, pipeline)
    pipe.run()

    # FULL cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
