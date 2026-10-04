import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize


def wall_metrics(predicted: np.ndarray, target: np.ndarray, tolerance: int = 2) -> dict:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("prediction and target must have the same 2D shape")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    pred, truth = predicted > 0, target > 0
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    empty = not (tp + fp + fn)
    p = tp / (tp + fp) if tp + fp else float(empty)
    r = tp / (tp + fn) if tp + fn else float(empty)
    pred_skeleton, true_skeleton = skeletonize(pred), skeletonize(truth)
    expand = (lambda x: binary_dilation(x, iterations=tolerance)) if tolerance else lambda x: x
    sk_p = float((pred_skeleton & expand(truth)).sum() / max(int(pred_skeleton.sum()), 1))
    sk_r = float((true_skeleton & expand(pred)).sum() / max(int(true_skeleton.sum()), 1))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / (tp + fp + fn) if not empty else 1.0,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "skeleton_precision": sk_p if not empty else 1.0,
        "skeleton_recall": sk_r if not empty else 1.0,
        "skeleton_f1": 2 * sk_p * sk_r / (sk_p + sk_r) if sk_p + sk_r else float(empty),
    }
