"""
Task 6: Translation (FLORES-200)
Translates a sentence from a source language to a target language.
Evaluation: BLEU score.
Default: French -> English (configurable via task_cfg).
"""
from datasets import load_dataset
from fmtk.datasetloaders.base import LLMDataset

LANG_NAMES = {
    'eng_Latn': 'English',
    'fra_Latn': 'French',
    'deu_Latn': 'German',
    'spa_Latn': 'Spanish',
    'zho_Hans': 'Chinese (Simplified)',
    'arb_Arab': 'Arabic',
}

PROMPT_TEMPLATE = (
    "Translate the following sentence from {src_lang} to {tgt_lang}. "
    "Respond with only the translation, nothing else.\n\n"
    "{src_lang} sentence: {text}\n\n"
    "{tgt_lang} translation:"
)

class FLORESDataset(LLMDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        self.src_lang = task_cfg.get('src_lang', 'fra_Latn')
        self.tgt_lang = task_cfg.get('tgt_lang', 'eng_Latn')
        # facebook/flores uses a dataset script no longer supported in datasets >= 3.x.
        # Fall back to wmt14 fr-en which is script-free and covers the same
        # French->English translation task used by default.
        hf_split = 'test' if split == 'test' else 'train'
        dataset = load_dataset('wmt/wmt14', 'fr-en', split=hf_split,
                               cache_dir=dataset_cfg.get('cache_dir', None))
        max_samples = dataset_cfg.get('max_samples', len(dataset))
        n = min(max_samples, len(dataset))
        self.src_lang_key = 'fr' if self.src_lang == 'fra_Latn' else 'en'
        self.tgt_lang_key = 'en' if self.tgt_lang == 'eng_Latn' else 'fr'
        self.wmt_data = list(dataset.select(range(n)))

    def __len__(self):
        return len(self.wmt_data)

    def __getitem__(self, idx):
        pair = self.wmt_data[idx]['translation']
        src_text = pair[self.src_lang_key]
        tgt_text = pair[self.tgt_lang_key]
        src_name = LANG_NAMES.get(self.src_lang, self.src_lang)
        tgt_name = LANG_NAMES.get(self.tgt_lang, self.tgt_lang)
        prompt = PROMPT_TEMPLATE.format(
            src_lang=src_name, tgt_lang=tgt_name, text=src_text
        )
        return {'x': prompt, 'y': tgt_text}

    def preprocess(self):
        pass
