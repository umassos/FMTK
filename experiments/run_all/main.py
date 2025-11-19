import subprocess
import json
from config import tasks

if __name__ == "__main__":
    for task_name, task_info in tasks.items():
        print(f"\n=== Running task: {task_name} ===")

        for p in task_info["pipelines"]:
            backbone = p["backbone"]
            dataset  = task_info["datasets"][0]

            print(f"--- Running model {backbone} on dataset {dataset} ---")

            subprocess.run([
                "python3",
                "worker.py",
                json.dumps({
                    "task_name": task_name,
                    "task_info": task_info,
                    "pipeline": p
                })
            ])
