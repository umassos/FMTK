# Quick sanity check - run this to verify pipeline
from rfdetr import RFDETRNano
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.misc import collate_fn
import torch

device = "cuda"
rfdetr = RFDETRNano(device=device)
config = rfdetr.model_config
transforms = make_coco_transforms_square_div_64(
    image_set="val", resolution=config.resolution,
    patch_size=config.patch_size, num_windows=config.num_windows,
)
val_dataset = CocoDetection("/datasets/ai/coco/val2017", "/datasets/ai/coco/annotations/instances_val2017.json", transforms=transforms)
data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)
base_ds = get_coco_api_from_dataset(val_dataset)
criterion, postprocess = build_criterion_and_postprocessors(rfdetr.model.args)
rfdetr.model.model.eval()
with torch.no_grad():
    stats, _ = evaluate(rfdetr.model.model, criterion, postprocess, data_loader, base_ds, torch.device(device), args=rfdetr.model.args)
print(stats["results_json"])  # Should show mAP@50 ~0.67