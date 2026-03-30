"""
Task 5: Summarization (CNN / DailyMail)
Model generates a short summary of a news article.
Evaluation: ROUGE-1, ROUGE-2, ROUGE-L.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

PROMPT_TEMPLATE = (
    "Write a concise summary of the following news article in 2-3 sentences.\n\n"
    "Article: {article}\n\n"
    "Summary:"
)

class CNNDailyMailDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        hf_split = 'test' if split == 'test' else 'train'
        dataset = load_dataset('abisee/cnn_dailymail', '3.0.0', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article = item['article'][:2000]
        highlights = item['highlights']
        prompt = PROMPT_TEMPLATE.format(article=article)
        return {'question': prompt, 'y': highlights}

    def preprocess(self):
        pass
