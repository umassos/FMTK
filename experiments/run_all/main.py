from config import *
from timeseries.experiments.run_all.inference_pipeline import InferencePipeline
import csv
import os
import json
if __name__ == "__main__":
    for task_name,task_info in tasks.items():
        print(f"Running task: {task_name}")
        for p in task_info['pipelines']:
            print(f"Running inference for model: {p['backbone']} on dataset: {task_info['datasets'][0]}")
            pipeline = InferencePipeline(task_name,task_info, p)
            pipeline.run()
 
            #save results to a file
            # results_file = f"results_{model_name}_{task_name}.json"
            # with open(results_file, "w") as f:
            #     json.dump(results, f, indent=4)
            # print(f"Results for {model_name} on {task_name}")
                    