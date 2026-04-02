"""
Task 1: Text Classification (AG News)
4 classes: World, Sports, Business, Sci/Tech
Uses HuggingFace datasets library.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

LABEL_MAP = {0: 'world', 1: 'sports', 2: 'business', 3: 'sci/tech'}

PROMPT_TEMPLATE = (
    "Classify the following news article into exactly one of these categories: "
    "World, Sports, Business, Sci/Tech.\n\n"
    "Article: {text}\n\n"
    "Respond with only the category name, nothing else."
)

class AGNewsDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        hf_split = 'test' if split == 'test' else 'train'
        dataset = load_dataset('ag_news', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = LABEL_MAP[item['label']]
        prompt = PROMPT_TEMPLATE.format(text=text[:1000])
        return {'question': prompt, 'y': label}

    def preprocess(self):
        pass
