"""
Task 7: Math Reasoning (GSM8K)
Grade-school math word problems requiring multi-step reasoning.
Evaluation: exact match on the final numeric answer.
"""
import re
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end, write the final numeric answer on a new line starting with 'Answer:'.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)

def extract_answer(text):
    """Extract the numeric answer from GSM8K gold answer string (#### number)."""
    match = re.search(r'####\s*([\d,\-\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return text.strip()

class GSM8KDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        hf_split = 'test' if split == 'test' else 'train'
        dataset = load_dataset('openai/gsm8k', 'main', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        # Gold label: just the numeric answer
        gold_answer = extract_answer(item['answer'])
        prompt = PROMPT_TEMPLATE.format(question=question)
        return {'x': prompt, 'y': gold_answer}

    def preprocess(self):
        pass
