"""
Task 8: Code Generation (HumanEval)
Model completes a Python function given a docstring.
Evaluation: pass@1 (functional correctness via execution).
Note: Execution-based eval requires running generated code safely.
      For safety, we measure exact-match / BLEU against canonical solution by default.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

PROMPT_TEMPLATE = (
    "Complete the following Python function. "
    "Write only the function body (the code that goes inside the function), "
    "no explanations.\n\n"
    "{prompt}"
)

class HumanEvalDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        # HumanEval only has a 'test' split
        dataset = load_dataset('openai/openai_humaneval', split='test',
                               cache_dir=dataset_cfg.get('cache_dir', None),
                               trust_remote_code=True)
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = PROMPT_TEMPLATE.format(prompt=item['prompt'])
        # canonical_solution is the gold reference
        return {'question': prompt, 'y': item['canonical_solution']}

    def preprocess(self):
        pass
