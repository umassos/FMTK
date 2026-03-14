"""
Task 3: Named Entity Recognition (CoNLL-2003)
Model outputs a JSON list of (entity, type) pairs.
Evaluation: token-level F1 via seqeval.
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

# CoNLL-2003 NER tag mapping
ID2TAG = {
    0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
    5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
}

PROMPT_TEMPLATE = (
    "Extract all named entities from the following sentence. "
    "For each entity, provide the entity text and its type "
    "(PER for person, ORG for organization, LOC for location, MISC for miscellaneous).\n\n"
    "Sentence: {sentence}\n\n"
    "Respond with a JSON list like: "
    '[{{"entity": "John", "type": "PER"}}, {{"entity": "Google", "type": "ORG"}}]. '
    "If no entities, respond with []."
)

class CoNLL2003Dataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        hf_split = split  # train / validation / test
        # 'conll2003' uses a dataset script no longer supported in datasets >= 3.x.
        # tner/conll2003 has the same data as plain JSON files (no script).
        split_file = {"train": "train", "validation": "valid", "test": "test"}[hf_split]
        dataset = load_dataset(
            "json",
            data_files=f"hf://datasets/tner/conll2003/dataset/{split_file}.json",
            split="train",
            cache_dir=dataset_cfg.get('cache_dir', None),
        )
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        self.data = list(dataset.select(range(min(max_samples, len(dataset)))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        ner_tags = item.get('ner_tags') or item.get('tags', [])
        sentence = ' '.join(tokens)

        # Build gold answer as BIO tag string for evaluation
        gold_tags = ' '.join(ID2TAG[t] for t in ner_tags)

        prompt = PROMPT_TEMPLATE.format(sentence=sentence)
        return {'x': prompt, 'y': gold_tags, 'tokens': tokens}

    def preprocess(self):
        pass
