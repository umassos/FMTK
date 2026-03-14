import re
import numpy as np
from collections import Counter
from sklearn.metrics import mean_absolute_error, accuracy_score

def get_mae(y_test, y_pred):
    if len(y_test.shape) > 2:
        y_test = y_test.reshape(-1, y_test.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    return mean_absolute_error(y_test, y_pred)

def get_accuracy(y_test, y_pred):
    def normalize(x):
        if isinstance(x, str):
            return x.lower()
        return x

    y_test = [normalize(y) for y in y_test]
    y_pred = [normalize(y) for y in y_pred]

    return accuracy_score(y_test, y_pred)

# ── LLM metrics ────────────────────────────────────────────────────────────────

def _normalize_text(text):
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def get_exact_match(y_test, y_pred):
    """
    Exact match after lowercasing and stripping.
    Used for: QA (SQuAD), math (GSM8K), multiple-choice (HellaSwag).
    """
    matches = sum(
        t.strip().lower() == p.strip().lower()
        for t, p in zip(y_test, y_pred)
    )
    return matches / len(y_test) if len(y_test) > 0 else 0.0

def get_token_f1(y_test, y_pred):
    """
    Token-level F1 (SQuAD style). Useful for QA tasks.
    """
    def _f1(gold, pred):
        gold_toks = _normalize_text(gold)
        pred_toks = _normalize_text(pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        return 2 * precision * recall / (precision + recall)

    scores = [_f1(g, p) for g, p in zip(y_test, y_pred)]
    return float(np.mean(scores))

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def _bleu_sentence(reference_tokens, hypothesis_tokens, max_n=4):
    import math
    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = _ngram_counts(reference_tokens, n)
        hyp_ngrams = _ngram_counts(hypothesis_tokens, n)
        clipped = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
        total = max(len(hypothesis_tokens) - n + 1, 0)
        scores.append(clipped / total if total > 0 else 0.0)
    if all(s == 0 for s in scores):
        return 0.0
    log_avg = sum(math.log(s) for s in scores if s > 0) / max_n
    bp = min(1.0, len(hypothesis_tokens) / max(len(reference_tokens), 1))
    return bp * math.exp(log_avg)

def get_bleu(y_test, y_pred):
    """
    Corpus-level BLEU-4. Used for: translation (FLORES), code generation (HumanEval).
    """
    scores = [
        _bleu_sentence(_normalize_text(ref), _normalize_text(hyp))
        for ref, hyp in zip(y_test, y_pred)
    ]
    return float(np.mean(scores))

def _rouge_n(reference_tokens, hypothesis_tokens, n):
    ref_ngrams = _ngram_counts(reference_tokens, n)
    hyp_ngrams = _ngram_counts(hypothesis_tokens, n)
    overlap = sum(min(c, hyp_ngrams[ng]) for ng, c in ref_ngrams.items())
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    precision = overlap / max(sum(hyp_ngrams.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _rouge_l(reference_tokens, hypothesis_tokens):
    m, n = len(reference_tokens), len(hypothesis_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference_tokens[i-1] == hypothesis_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def get_rouge(y_test, y_pred):
    """
    Returns dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    Used for: summarization (CNN/DailyMail).
    """
    r1, r2, rl = [], [], []
    for ref, hyp in zip(y_test, y_pred):
        ref_toks = _normalize_text(ref)
        hyp_toks = _normalize_text(hyp)
        r1.append(_rouge_n(ref_toks, hyp_toks, 1))
        r2.append(_rouge_n(ref_toks, hyp_toks, 2))
        rl.append(_rouge_l(ref_toks, hyp_toks))
    return {
        'rouge1': float(np.mean(r1)),
        'rouge2': float(np.mean(r2)),
        'rougeL': float(np.mean(rl)),
    }

def get_gsm8k_accuracy(y_test, y_pred):
    """
    Extract final numeric answer from model output and compare to gold.
    Used for: GSM8K math reasoning.
    """
    def extract_number(text):
        match = re.search(r'[Aa]nswer\s*[:\-]?\s*([\-\d,\.]+)', text)
        if match:
            return match.group(1).replace(',', '').strip()
        numbers = re.findall(r'[\-\d,\.]+', text)
        return numbers[-1].replace(',', '') if numbers else ''

    matches = sum(
        extract_number(p) == g.strip()
        for g, p in zip(y_test, y_pred)
    )
    return matches / len(y_test) if len(y_test) > 0 else 0.0