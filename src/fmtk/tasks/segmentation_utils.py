import math
import torch
import torch.nn.functional as F


def center_pad(x, multiple):
    # Zero-pad H, W up to a multiple of the patch size, split evenly (DINOv2 CenterPadding)
    h, w = x.shape[-2:]
    ph = math.ceil(h / multiple) * multiple - h
    pw = math.ceil(w / multiple) * multiple - w
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def slide_inference(P, img, out_size, num_classes, crop_size, stride, patch_size, device):
    """
    mmseg-style sliding-window inference.
    img: [1, 3, H, W]; returns logits [1, num_classes, *out_size].
    The decoder must upsample to (crop_size, crop_size).
    """
    _, _, H, W = img.shape
    h_grids = max(H - crop_size + stride - 1, 0) // stride + 1
    w_grids = max(W - crop_size + stride - 1, 0) // stride + 1

    crops, boxes = [], []
    for i in range(h_grids):
        for j in range(w_grids):
            y2, x2 = min(i * stride + crop_size, H), min(j * stride + crop_size, W)
            y1, x1 = max(y2 - crop_size, 0), max(x2 - crop_size, 0)
            crops.append(img[:, :, y1:y2, x1:x2])
            boxes.append((y1, y2, x1, x2))

    crop_logits = P.forward(center_pad(torch.cat(crops), patch_size).to(device))

    logits = torch.zeros(1, num_classes, H, W, device=crop_logits.device)
    count = torch.zeros(1, 1, H, W, device=crop_logits.device)
    for (y1, y2, x1, x2), crop_logit in zip(boxes, crop_logits):
        logits[:, :, y1:y2, x1:x2] += crop_logit
        count[:, :, y1:y2, x1:x2] += 1
    logits = logits / count

    return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
