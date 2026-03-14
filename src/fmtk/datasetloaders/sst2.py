"""
Task 2: Sentiment Analysis (SST-2)
Binary: positive / negative.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

LABEL_MAP = {0: 'negative', 1: 'positive'}

PROMPT_TEMPLATE = (
    "Analyze the sentiment of the following sentence and respond with exactly "
    "one word: positive or negative.\n\n"
    "Sentence: {text}\n\n"
    "Sentiment:"
)

class SST2Dataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        hf_split = 'validation' if split == 'test' else 'train'
        dataset = load_dataset('glue', 'sst2', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        # SST-2 validation has labels; filter out unlabelled test set entries
        dataset = dataset.filter(lambda x: x['label'] != -1)
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = LABEL_MAP[item['label']]
        prompt = PROMPT_TEMPLATE.format(text=item['sentence'])
        return {'x': prompt, 'y': label}

    def preprocess(self):
        pass
