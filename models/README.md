# Models

This directory stores all model weights organized by model family.

## Structure

```
models/
├── tsfm/                   # Time Series Foundation Models
│   ├── pretrained/         # Pre-trained weights (.pt files)
│   │   ├── papagei_p.pt
│   │   ├── papagei_s.pt
│   │   └── papagei_s_svri.pt
│   └── finetuned/          # Fine-tuned checkpoints, one folder per run
│       └── {task}_{backbone}_{decoder}[_{encoder}]_[{adapter}]/
│           ├── decoder.pth
│           ├── encoder.pth     # if encoder was trained
│           ├── adapter.pth     # if LoRA adapter was trained
│           └── pipeline.json   # timing and resource metrics of finetuning time
├── vlms/                   # Vision-Language Models
│   ├── pretrained/         # HuggingFace cache (llava, qwen, molmo, etc.)
│   └── finetuned/          # Fine-tuned VLM checkpoints
└── vision/                 # Vision Models
    ├── pretrained/         # HuggingFace/torchvision cache (dinov2, swin, vgg, mae)
    └── finetuned/          # Fine-tuned vision checkpoints
```

## Run Naming Convention

Finetuned runs follow the pattern:

```
{task}_{backbone}_{decoder}[_{encoder}]_[{adapter}]
```

| Segment    | Examples                                        |
|------------|-------------------------------------------------|
| `task`     | `diasbp`, `heartrate`, `ecgclass`, `etth1_fore` |
| `backbone` | `chronosbase`, `momentlarge`, `papageis`         |
| `decoder`  | `mlp`, `linear`, `lstm`                         |
|`encoder`   |  `mlp`|
| `adapter`  | `lora`               |

Example: `diasbp_chronosbase_mlp`, `etth1_fore_momentsmall_mlp_lora`
