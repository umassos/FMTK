"""
Task 4: Question Answering (SQuAD v1.1)
Model reads a passage and answers a question in free text.
Evaluation: exact match and F1 over answer tokens.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

PROMPT_TEMPLATE = (
    "Read the following passage carefully and answer the question based only on the passage.\n\n"
    "Passage: {context}\n\n"
    "Question: {question}\n\n"
    "Answer (be concise, copy words from the passage where possible):"
)

class SQuADDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        hf_split = 'validation' if split == 'test' else 'train'
        dataset = load_dataset('rajpurkar/squad', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context'][:1500]
        question = item['question']
        # Gold answer: first answer text
        answer = item['answers']['text'][0] if item['answers']['text'] else ''
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        return {'question': prompt, 'y': answer}

    def preprocess(self):
        pass
