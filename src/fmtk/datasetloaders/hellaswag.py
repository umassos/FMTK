"""
Task 10: Reading Comprehension / Commonsense (HellaSwag)
4-choice multiple choice: pick the most plausible sentence continuation.
Evaluation: accuracy.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

PROMPT_TEMPLATE = (
    "Choose the most plausible continuation for the following activity description. "
    "Respond with only the letter (A, B, C, or D).\n\n"
    "Activity: {activity_label}\n"
    "Context: {ctx}\n\n"
    "Options:\n"
    "A) {choice_a}\n"
    "B) {choice_b}\n"
    "C) {choice_c}\n"
    "D) {choice_d}\n\n"
    "Answer:"
)

CHOICE_LABELS = ['A', 'B', 'C', 'D']

class HellaSwagDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        hf_split = 'validation' if split == 'test' else 'train'
        dataset = load_dataset('Rowan/hellaswag', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        endings = item['endings']  # list of 4 strings
        gold_label = CHOICE_LABELS[int(item['label'])]
        prompt = PROMPT_TEMPLATE.format(
            activity_label=item['activity_label'],
            ctx=item['ctx'],
            choice_a=endings[0],
            choice_b=endings[1],
            choice_c=endings[2],
            choice_d=endings[3],
        )
        return {'question': prompt, 'y': gold_label}

    def preprocess(self):
        pass
