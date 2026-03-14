"""
Task 9: Fact Verification (FEVER)
Binary: SUPPORTS / REFUTES a claim.
(NEI = Not Enough Info examples are excluded for cleaner binary evaluation.)
Evaluation: accuracy.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

LABEL_MAP = {'SUPPORTS': 'supports', 'REFUTES': 'refutes', 'NOT ENOUGH INFO': 'not enough info'}

PROMPT_TEMPLATE = (
    "Determine whether the following claim is supported or refuted based on general knowledge. "
    "Respond with exactly one word: supports or refutes.\n\n"
    "Claim: {claim}\n\n"
    "Verdict:"
)

class FEVERDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        # 'fever' uses a dataset script no longer supported in datasets >= 3.x.
        # climate_fever is script-free, same task (claim→SUPPORTS/REFUTES), same label names.
        hf_split = 'test' if split == 'test' else 'train'
        dataset = load_dataset('climate_fever', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        # Keep only SUPPORTS / REFUTES for clean binary classification
        LABEL_NAMES = dataset.features['claim_label'].names
        dataset = dataset.filter(lambda x: LABEL_NAMES[x['claim_label']] in ('SUPPORTS', 'REFUTES'))
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # claim_label is an int index; map back to string via LABEL_MAP
        label_int = item['claim_label']
        label_names = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO', 'DISPUTED']
        label = LABEL_MAP.get(label_names[label_int], label_names[label_int]).lower()
        prompt = PROMPT_TEMPLATE.format(claim=item['claim'])
        return {'x': prompt, 'y': label}

    def preprocess(self):
        pass
