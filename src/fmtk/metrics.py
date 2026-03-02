import numpy as np
import torch
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


def get_mIoU(y_true, y_pred, num_classes, ignore_index=255):
    """Compute mean Intersection-over-Union for semantic segmentation.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class map(s).  Shape ``(N, H, W)`` or ``(H, W)``.
        Each pixel value is a class index in ``[0, num_classes)`` or
        ``ignore_index``.
    y_pred : array-like
        Predicted class map(s).  Same shape as *y_true*.
        If shape is ``(N, C, H, W)`` (logits / probabilities), argmax
        over the class dimension is applied automatically.
    num_classes : int
        Total number of classes (e.g. 21 for Pascal VOC).
    ignore_index : int, optional
        Label value to ignore when computing IoU (default 255).

    Returns
    -------
    dict
        ``"mIoU"``  – scalar mean IoU across classes that appear in the
        ground truth (excluding *ignore_index*).
        ``"per_class_iou"`` – list of per-class IoU values (``float`` or
        ``None`` if the class is absent from both gt and pred).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # Handle logits: (N, C, H, W) -> (N, H, W)
    if y_pred.ndim == 4:
        y_pred = y_pred.argmax(axis=1)

    # Flatten to 1-D
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Mask out ignored pixels
    valid = y_true != ignore_index
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    # Build confusion matrix via bincount
    assert y_true.shape == y_pred.shape
    combined = y_true * num_classes + y_pred
    cm = np.bincount(combined, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)

    # Per-class IoU = TP / (TP + FP + FN)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn

    per_class_iou = []
    valid_ious = []
    for c in range(num_classes):
        if denom[c] == 0:
            per_class_iou.append(None)
        else:
            iou = float(tp[c] / denom[c])
            per_class_iou.append(iou)
            valid_ious.append(iou)

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {"mIoU": miou, "per_class_iou": per_class_iou}


class StreamingMIoU:
    """Accumulates a confusion matrix batch-by-batch so the full
    prediction arrays never need to be held in memory at once.

    Usage::

        meter = StreamingMIoU(num_classes=21, ignore_index=255)
        for y_true_batch, y_pred_batch in ...:
            meter.update(y_true_batch, y_pred_batch)
        result = meter.compute()   # {"mIoU": ..., "per_class_iou": ...}
    """

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

        if y_pred.ndim == 4:
            y_pred = y_pred.argmax(axis=1).ravel()

        valid = y_true != self.ignore_index
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        combined = y_true * self.num_classes + y_pred
        batch_cm = np.bincount(
            combined, minlength=self.num_classes * self.num_classes
        )
        self.cm += batch_cm.reshape(self.num_classes, self.num_classes)

    def compute(self):
        tp = np.diag(self.cm)
        fp = self.cm.sum(axis=0) - tp
        fn = self.cm.sum(axis=1) - tp
        denom = tp + fp + fn

        per_class_iou = []
        valid_ious = []
        for c in range(self.num_classes):
            if denom[c] == 0:
                per_class_iou.append(None)
            else:
                iou = float(tp[c] / denom[c])
                per_class_iou.append(iou)
                valid_ious.append(iou)

        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        return {"mIoU": miou, "per_class_iou": per_class_iou}

    def reset(self):
        self.cm[:] = 0