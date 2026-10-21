"""DART: Discriminative Axis Real-time Tracker.

DART keeps two prototypes per feature layer, one for ID and one for OOD, and
updates them online from unlabeled test batches. Each sample is scored by its
relative distance to the two prototypes (RDS), so the score follows the
discriminative axis ``p_I - p_O`` as it drifts under covariate shift.
"""

import logging

import numpy as np
import torch

from .utils import otsu_threshold, tukey_inlier_mask

logger = logging.getLogger(__name__)

# Minimum number of samples a pseudo-labeled group needs before outlier
# filtering (or a flip check) is applied to it.
MIN_GROUP_SIZE = 10


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class DART:
    """Dual-prototype tracker for a single feature layer.

    Args:
        ema_alpha: EMA coefficient for the prototype update (Eq. 4).
        temperature: softmax temperature of the MSP reference score.
        iqr_factor: Tukey fence multiplier for outlier filtering.
        flip_weight: distance ratio required to declare a prototype flip (Eq. 5).
        check_interval: run the flip check every ``check_interval`` batches.
        flip: enable flip correction. ``False`` gives DART-NoFlip.
    """

    def __init__(self, ema_alpha=0.8, temperature=5.0, iqr_factor=1.5,
                 flip_weight=2.0, check_interval=10, flip=True):
        self.alpha = ema_alpha
        self.temperature = temperature
        self.iqr_factor = iqr_factor
        self.flip_weight = flip_weight
        self.check_interval = check_interval
        self.flip = flip

        self.id_proto = None
        self.ood_proto = None
        self.num_batches = 0  # batches processed after initialization
        self.flip_history = []

    @property
    def initialized(self):
        return self.id_proto is not None

    def __call__(self, features, logits):
        """Score one test batch and update the prototypes.

        Args:
            features: (N, D) features of this layer.
            logits: (N, C) classifier logits, used for the MSP reference.

        Returns:
            np.ndarray of shape (N,). Higher means more likely ID.
        """
        features = _to_numpy(features)
        logits = torch.as_tensor(_to_numpy(logits))

        if not self.initialized:
            self._initialize(features, logits)
        else:
            self.num_batches += 1
            if self.flip and self.num_batches % self.check_interval == 0:
                self._check_flip(features, logits)

        rds, _, _ = self.compute_rds(features)
        is_id = rds < otsu_threshold(rds)
        self._update(features, is_id)
        return -rds

    def compute_rds(self, features):
        """Relative Distance Score (Eq. 3). Lower means closer to ID."""
        d_id = np.linalg.norm(features - self.id_proto, axis=1)
        d_ood = np.linalg.norm(features - self.ood_proto, axis=1)
        return d_id / (d_id + d_ood + 1e-10), d_id, d_ood

    # ------------------------------------------------------------------
    # MSP reference
    # ------------------------------------------------------------------
    def _msp_pseudo_labels(self, features, logits):
        """Pseudo-label a batch with MSP + Otsu, ignoring feature outliers.

        Returns (is_id, inlier) boolean masks. Outliers are samples whose
        distance to the batch mean exceeds the Tukey fence.
        """
        msp = torch.softmax(logits / self.temperature, dim=1).max(dim=1).values.numpy()

        dist_to_mean = np.linalg.norm(features - features.mean(axis=0), axis=1)
        q1, q3 = np.percentile(dist_to_mean, [25, 75])
        inlier = dist_to_mean <= q3 + self.iqr_factor * (q3 - q1)

        scores_for_threshold = msp[inlier] if inlier.sum() > MIN_GROUP_SIZE else msp
        is_id = (msp > otsu_threshold(scores_for_threshold)) & inlier
        return is_id, inlier

    def _initialize(self, features, logits):
        """Initialize both prototypes from the first batch (Sec. 3.3)."""
        is_id, inlier = self._msp_pseudo_labels(features, logits)
        id_mask, ood_mask = is_id, ~is_id & inlier

        if not (id_mask.any() and ood_mask.any()):
            # Retry with outliers counted as OOD.
            id_mask, ood_mask = is_id, ~is_id
        if id_mask.any() and ood_mask.any():
            self.id_proto = features[id_mask].mean(axis=0)
            self.ood_proto = features[ood_mask].mean(axis=0)
        else:
            mid = len(features) // 2
            self.id_proto = features[:mid].mean(axis=0)
            self.ood_proto = features[mid:].mean(axis=0)
            logger.warning("MSP pseudo-labeling failed; initialized by splitting the batch in half.")

    def _check_flip(self, features, logits):
        """Swap the prototypes if they have drifted to opposite sides (Eq. 5)."""
        is_id, inlier = self._msp_pseudo_labels(features, logits)
        id_mask, ood_mask = is_id, ~is_id & inlier
        if id_mask.sum() < MIN_GROUP_SIZE or ood_mask.sum() < MIN_GROUP_SIZE:
            return False

        ref_id = features[id_mask].mean(axis=0)

        def cos_dist(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        far_in_l2 = (np.linalg.norm(self.id_proto - ref_id)
                     > self.flip_weight * np.linalg.norm(self.ood_proto - ref_id))
        far_in_cos = cos_dist(self.id_proto, ref_id) > cos_dist(self.ood_proto, ref_id)

        if far_in_l2 and far_in_cos:
            self.id_proto, self.ood_proto = self.ood_proto, self.id_proto
            self.flip_history.append(self.num_batches)
            logger.info(f"Prototype flip detected at batch {self.num_batches}")
            return True
        return False

    # ------------------------------------------------------------------
    # Prototype update
    # ------------------------------------------------------------------
    def _update(self, features, is_id):
        """Tukey-filter each pseudo-labeled group, then EMA-update (Eq. 4)."""
        self.id_proto = self._ema_step(self.id_proto, features[is_id])
        self.ood_proto = self._ema_step(self.ood_proto, features[~is_id])

    def _ema_step(self, proto, group):
        if len(group) == 0:
            return proto
        if len(group) > MIN_GROUP_SIZE:
            group = group[tukey_inlier_mask(np.linalg.norm(group - proto, axis=1), self.iqr_factor)]
        return self.alpha * proto + (1 - self.alpha) * group.mean(axis=0)


class MultiLayerDART:
    """DART over several layers, fused by averaging the per-layer scores (Sec. 3.4)."""

    def __init__(self, layers, **dart_kwargs):
        self.layers = list(layers)
        self.trackers = {layer: DART(**dart_kwargs) for layer in self.layers}

    def __call__(self, features_by_layer, logits):
        scores = [self.trackers[layer](features_by_layer[layer], logits) for layer in self.layers]
        return np.mean(scores, axis=0)
