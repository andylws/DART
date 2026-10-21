"""Online evaluation loop: feed mixed ID/OOD batches to DART and score them."""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from .features import extract_features

logger = logging.getLogger(__name__)


def fpr_at_95_tpr(labels, scores):
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    return fpr[np.argmin(np.abs(tpr - 0.95))]


def evaluate_stream(detector, backbone, x_ind, x_ood, batch_size=100):
    """Run DART over a stream and report AUROC / FPR@95TPR (ID is the positive class).

    Each batch holds the next ``batch_ind`` ID and ``batch_ood`` OOD samples,
    so every batch keeps the global ID:OOD ratio.

    Returns:
        dict with pooled metrics over the whole stream (``auroc``, ``fpr95``),
        batch-averaged metrics (``avg_auroc``, ``avg_fpr95``) and the per-batch
        lists.
    """
    n_ind, n_ood = x_ind.size(0), x_ood.size(0)
    num_batches = max(math.ceil(n_ind / batch_size), math.ceil(n_ood / batch_size))
    batch_ind = min(math.ceil(n_ind / num_batches), batch_size)
    batch_ood = min(math.ceil(n_ood / num_batches), batch_size)

    all_scores, all_labels, batch_aurocs, batch_fprs = [], [], [], []
    for b in range(num_batches):
        xi = x_ind[b * batch_ind:(b + 1) * batch_ind]
        xo = x_ood[b * batch_ood:(b + 1) * batch_ood]
        x = torch.cat([xi, xo])
        if backbone.input_size is not None:
            x = F.interpolate(x, size=(backbone.input_size, backbone.input_size), mode="bilinear")
        labels = np.concatenate([np.ones(len(xi)), np.zeros(len(xo))])

        features, logits = extract_features(backbone.model, x, backbone.layers)
        scores = detector(features, logits)

        all_scores.append(scores)
        all_labels.append(labels)
        if 0 < labels.sum() < len(labels):
            batch_aurocs.append(metrics.roc_auc_score(labels, scores))
            batch_fprs.append(fpr_at_95_tpr(labels, scores))
            logger.debug(f"batch {b + 1}/{num_batches}: AUROC={batch_aurocs[-1]:.4f} "
                         f"FPR95={batch_fprs[-1]:.4f}")

    all_scores, all_labels = np.concatenate(all_scores), np.concatenate(all_labels)
    return {
        "auroc": metrics.roc_auc_score(all_labels, all_scores),
        "fpr95": fpr_at_95_tpr(all_labels, all_scores),
        "avg_auroc": float(np.mean(batch_aurocs)),
        "avg_fpr95": float(np.mean(batch_fprs)),
        "batch_aurocs": batch_aurocs,
        "batch_fprs": batch_fprs,
    }
