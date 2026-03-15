"""
VLM task definitions for FMTK.

Each task has:
  - prompt:    the text sent to the model (use {question} placeholder for VQA)
  - parser:    name of the parser function to clean raw model output
  - evaluator: name of the evaluator function to score predicted vs ground truth

Dataset paths:
  - Unified pipeline: experiments/run_all/config.py  (datasets{} with 'vlm_*' keys)
  - Standalone scripts: use get_vlm_dataset_config(task_name) from this module.
  All VLM data is sourced from FMTK/dataset/vlm/.
"""
import os
import re
from pathlib import Path

# ==================== TASK REGISTRY ====================
# prompt    : text prompt sent to the model.
#             VQA is the only task with a {question} placeholder;
#             all other tasks use a fixed prompt regardless of the sample.
# parser    : name string — resolved via get_parser()
# evaluator : name string — resolved via get_evaluator()

TASK_REGISTRY = {
    "crowd": {
        "prompt": (
            "Look at this image and estimate the crowd density. "
            "Answer with ONLY ONE of these categories: very_sparse, sparse, moderate, dense, or very_dense. "
            "Do not provide any explanation or numbers, just the category name. "
            "Please answer in a few words."
        ),
        "parser": "parse_crowd_label",
        "evaluator": "evaluate_crowd",
    },

    "scene": {
        "prompt": (
            "What type of scene is shown in this image? "
            "Answer with a short scene description (for example: 'kitchen', 'beach', 'office', 'mountain', etc.). "
            "Provide only the scene type, no additional explanation. "
            "Please answer in a few words."
        ),
        "parser": "parse_scene_label",
        "evaluator": "evaluate_scene",
    },

    "ocr": {
        "prompt": (
            "You are given an image of a single handwritten digit. "
            "Answer with exactly one character that is a digit from 0 to 9. "
            "Do not include any extra words."
            "Please answer in one word."
        ),
        "parser": "parse_ocr_digit",
        "evaluator": "evaluate_ocr",
    },

    # VQA is the only task where the prompt is built per-sample:
    # the {question} placeholder is replaced with the question from the JSON record.
    "vqa": {
        "prompt": "{question} Please answer in one word.",
        "parser": "parse_vqa_label",
        "evaluator": "evaluate_vqa",
    },

    "traffic": {
        "prompt": (
            "What traffic sign is shown in this image? "
            "Answer with the sign type only (e.g., 'stop sign', 'speed limit', 'yield', etc.). "
            "Please answer in a few words."
        ),
        "parser": "parse_traffic_label",
        "evaluator": "evaluate_substring_match",
    },

    "gesture": {
        "prompt": (
            "What hand gesture is being shown in this image? "
            "Answer with the gesture name only, without any explanation. "
            "Use simple terms like 'thumbs up', 'peace', 'stop', 'ok', 'fist', 'call', etc. "
            "Please answer in a few words."
        ),
        "parser": "parse_gesture_label",
        "evaluator": "evaluate_substring_match",
    },

    "activity": {
        "prompt": (
            "What activity is the person doing in this image? "
            "Answer with the activity name only, without any explanation. "
            "Choose from: calling, clapping, cycling, dancing, drinking, eating, fighting, "
            "hugging, laughing, listening to music, running, sitting, sleeping, texting, or using laptop."
        ),
        "parser": "parse_activity_label",
        "evaluator": "evaluate_substring_match",
    },

    "object_detection": {
        "prompt": (
            "What is the main object in this image? "
            "Answer with one object name from the COCO dataset. "
            "Provide only the object name without any explanation. "
        ),
        "parser": "parse_object_detection_label",
        "evaluator": "evaluate_object_detection",
    },

    "image_classification": {
        "prompt": (
            "What is the main object or subject in this image? "
            "Answer with one class name from ImageNet-1k. "
            "Provide only the class name without any explanation."
        ),
        "parser": "parse_classification_label",
        "evaluator": "evaluate_image_classification",
    },
}


# ==================== LABEL PARSERS ====================

def parse_crowd_label(text: str) -> str:
    """
    Parse crowd density category.
    Priority order matters: check very_dense before dense, very_sparse before sparse
    to avoid 'very_dense' being matched as just 'dense'.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    text = ' '.join(text.split())
    text = text.replace(' ', '_').replace('-', '_')

    if 'very_dense' in text or 'verydense' in text:
        return 'very_dense'
    elif 'very_sparse' in text or 'verysparse' in text:
        return 'very_sparse'
    elif 'dense' in text:
        return 'dense'
    elif 'moderate' in text:
        return 'moderate'
    elif 'sparse' in text:
        return 'sparse'

    return text


def parse_scene_label(text: str) -> str:
    """
    Parse scene classification label.
    Strips common model-generated prefixes and leading articles.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    text = ' '.join(text.split())

    prefixes = [
        "the scene is", "this is", "this looks like", "this appears to be",
        "the image shows", "it is", "it's", "scene:", "type:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    if text.startswith("a "):
        text = text[2:]
    elif text.startswith("an "):
        text = text[3:]
    elif text.startswith("the "):
        text = text[4:]

    return text


# Word-to-digit map used by parse_ocr_digit
_WORD2DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def parse_ocr_digit(text: str) -> str:
    """
    Parse a single handwritten digit from model output.
    First tries regex for a digit character, then word-to-digit mapping.
    """
    if not text:
        return ""
    t = text.strip().lower()
    m = re.search(r"\b([0-9])\b", t)
    if m:
        return m.group(1)
    for word, digit in _WORD2DIGIT.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return digit
    return ""


def parse_ocr_label(text: str) -> str:
    """Generic OCR text parser — whitespace normalisation only."""
    if not text:
        return ""
    return ' '.join(text.strip().split())


def parse_vqa_label(text: str) -> str:
    """
    Parse VQA answer.
    Lowercase only — no punctuation removal, matching the original workflow.
    """
    if not text:
        return ""
    return text.lower()


def parse_traffic_label(text: str) -> str:
    """Parse traffic sign label — strip, lowercase, punctuation removal."""
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    text = ' '.join(text.split())
    return text


def parse_gesture_label(text: str) -> str:
    """Parse gesture label — strip, lowercase, punctuation removal."""
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    text = ' '.join(text.split())
    return text


def parse_activity_label(text: str) -> str:
    """Parse activity label — strip, lowercase, punctuation removal."""
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')
    text = ' '.join(text.split())
    return text


def parse_classification_label(text: str) -> str:
    """
    Parse ImageNet classification label.
    Removes common model-generated prefixes before the class name.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = text.strip('.,!?;:"\'"')

    prefixes = ["this is", "the image shows", "it is", "it's", "a ", "an ", "the "]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return ' '.join(text.split())


def parse_object_detection_label(text: str) -> str:
    """
    Parse COCO object detection output.
    Takes only the first word — models often generate extra description.
    """
    if not text:
        return ""
    text = text.strip().lower()
    category = text.split()[0] if text.split() else ""
    category = category.strip('.,!?;:"\'"')
    return category


# ==================== EVALUATORS ====================

def evaluate_crowd(predicted: str, ground_truth: str) -> bool:
    """Bidirectional substring match for crowd density."""
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    return (gt in pred) or (pred in gt)


def evaluate_scene(predicted: str, ground_truth: str) -> bool:
    """Bidirectional substring match for scene classification."""
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    return (gt in pred) or (pred in gt)


def evaluate_ocr(predicted: str, ground_truth: str) -> bool:
    """Normalised exact match for OCR digits."""
    pred_norm = ' '.join(predicted.lower().strip().split())
    gt_norm = ' '.join(str(ground_truth).lower().strip().split())
    return pred_norm == gt_norm


def evaluate_vqa(predicted: str, ground_truth) -> bool:
    """
    VQA evaluation — substring match (gt in pred).
    Ground truth may be a single string or a list of valid answers;
    if a list, any answer matching is a correct prediction.
    """
    pred_norm = predicted.lower().strip()
    if isinstance(ground_truth, list):
        return any(ans.lower().strip() in pred_norm for ans in ground_truth)
    return str(ground_truth).lower().strip() in pred_norm


def evaluate_exact_match(predicted: str, ground_truth: str) -> bool:
    """Exact match after lowercasing and stripping."""
    return predicted.lower().strip() == str(ground_truth).lower().strip()


def evaluate_substring_match(predicted: str, ground_truth: str) -> bool:
    """
    Bidirectional substring match — used for traffic, gesture, activity.
    Correct if gt is in pred OR pred is in gt.
    """
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    return (gt in pred) or (pred in gt)


def evaluate_image_classification(predicted: str, ground_truth: str) -> bool:
    """
    One-way substring match for ImageNet classification.
    Correct if pred is a substring of gt (not the other way around).
    """
    pred = predicted.lower().strip()
    gt = str(ground_truth).lower().strip()
    return pred in gt


def evaluate_object_detection(predicted: str, ground_truth) -> bool:
    """
    COCO object detection evaluation.
    Ground truth is a list of valid category names for the image.
    Correct if the predicted category is in that list.
    """
    if not isinstance(ground_truth, (list, tuple)):
        ground_truth = [ground_truth]
    pred_category = predicted.lower().strip()
    gt_categories = [cat.lower().strip() for cat in ground_truth]
    return pred_category in gt_categories


# ==================== REGISTRY ACCESS ====================

def get_parser(parser_name: str):
    parsers = {
        "parse_crowd_label":          parse_crowd_label,
        "parse_scene_label":          parse_scene_label,
        "parse_ocr_digit":            parse_ocr_digit,
        "parse_ocr_label":            parse_ocr_label,
        "parse_vqa_label":            parse_vqa_label,
        "parse_traffic_label":        parse_traffic_label,
        "parse_gesture_label":        parse_gesture_label,
        "parse_activity_label":       parse_activity_label,
        "parse_classification_label": parse_classification_label,
        "parse_object_detection_label": parse_object_detection_label,
    }
    if parser_name not in parsers:
        raise ValueError(f"Unknown parser: '{parser_name}'. Available: {list(parsers)}")
    return parsers[parser_name]


def get_evaluator(evaluator_name: str):
    evaluators = {
        "evaluate_crowd":               evaluate_crowd,
        "evaluate_scene":               evaluate_scene,
        "evaluate_ocr":                 evaluate_ocr,
        "evaluate_vqa":                 evaluate_vqa,
        "evaluate_exact_match":         evaluate_exact_match,
        "evaluate_substring_match":     evaluate_substring_match,
        "evaluate_image_classification": evaluate_image_classification,
        "evaluate_object_detection":    evaluate_object_detection,
    }
    if evaluator_name not in evaluators:
        raise ValueError(f"Unknown evaluator: '{evaluator_name}'. Available: {list(evaluators)}")
    return evaluators[evaluator_name]


# ==================== SHARED DATASET CONFIGURATION ====================
# All VLM data lives under FMTK/dataset/vlm/.
# Resolve the shared root once (works regardless of working directory).

_FMTK_ROOT = Path(__file__).resolve().parents[3]          # .../FMTK
_VLM_DATA_ROOT = _FMTK_ROOT / "dataset" / "vlm"

VLM_DATASET_CONFIGS = {
    "activity": {
        "dataset_path": str(_VLM_DATA_ROOT / "activity_recognition"),
        "dataset_type": "vlm",
        "json_file":    "labels.json",
    },
    "crowd": {
        "dataset_path": str(_VLM_DATA_ROOT / "crowd_counting"),
        "dataset_type": "vlm",
        "json_file":    "labels.json",
    },
    "gesture": {
        "dataset_path": str(_VLM_DATA_ROOT / "gesture_recognition"),
        "dataset_type": "vlm",
        "json_file":    "labels.json",
    },
    "image_classification": {
        "dataset_path": str(_VLM_DATA_ROOT / "image_classification"),
        "dataset_type": "vlm",
        "json_file":    "labels.json",
    },
    "object_detection": {
        "dataset_path": str(_VLM_DATA_ROOT / "object_detection"),
        "dataset_type": "vlm",
        "json_file":    "annotations.json",
    },
    "ocr": {
        "dataset_path": str(_VLM_DATA_ROOT / "ocr"),
        "dataset_type": "vlm",
        "json_file":    "labels.json",
    },
    "scene": {
        "dataset_path": str(_VLM_DATA_ROOT / "scene_classification"),
        "dataset_type": "vlm",
        "json_file":    "labels.json",
    },
    "traffic": {
        "dataset_path": str(_VLM_DATA_ROOT / "traffic_classification"),
        "dataset_type": "vlm",
        "json_file":    "labels.json",
    },
    "vqa": {
        "dataset_path": str(_VLM_DATA_ROOT / "vqa"),
        "dataset_type": "vlm",
        "json_file":    "val.json",
        "image_subdir": "val2014",
    },
}


def get_vlm_dataset_config(task_name: str):
    """Return (dataset_cfg, task_cfg) for a VLM task using the shared data root.

    Example usage in a standalone profiling script::

        from fmtk.tasks.vlm_utils import get_vlm_dataset_config
        from fmtk.datasetloaders.vlm_dataset import VLMDataset

        dataset_cfg, task_cfg = get_vlm_dataset_config("scene")
        dataset = VLMDataset(dataset_cfg, task_cfg, split="test")
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown VLM task: '{task_name}'. "
            f"Available: {sorted(TASK_REGISTRY)}"
        )
    if task_name not in VLM_DATASET_CONFIGS:
        raise ValueError(
            f"No shared dataset config for task '{task_name}'. "
            f"Available: {sorted(VLM_DATASET_CONFIGS)}"
        )
    dataset_cfg = dict(VLM_DATASET_CONFIGS[task_name])
    task_cfg = {"prompt": TASK_REGISTRY[task_name]["prompt"]}
    return dataset_cfg, task_cfg
