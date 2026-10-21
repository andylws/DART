"""Thresholding and outlier-filtering helpers used by DART."""

import numpy as np


def otsu_threshold(scores, bins=50):
    """Return the Otsu threshold of a 1-D score array.

    The threshold maximizes the between-class variance of a ``bins``-bin
    histogram and is returned as the center of the selected bin.
    """
    hist, bin_edges = np.histogram(scores, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    weight1, weight2 = 0, total
    sum1, sum2 = 0.0, np.sum(hist * bin_centers)

    best_thresh, best_between_variance = 0, 0
    for i in range(len(hist)):
        weight1 += hist[i]
        weight2 -= hist[i]
        if weight1 == 0 or weight2 == 0:
            continue

        sum1 += bin_centers[i] * hist[i]
        sum2 -= bin_centers[i] * hist[i]
        mean1, mean2 = sum1 / weight1, sum2 / weight2

        between_variance = weight1 * weight2 * (mean1 - mean2) ** 2
        if between_variance > best_between_variance:
            best_between_variance = between_variance
            best_thresh = bin_centers[i]

    return best_thresh


def tukey_inlier_mask(distances, iqr_factor=1.5):
    """Tukey's fence on the upper side only: keep ``d <= Q3 + k * IQR``."""
    q1, q3 = np.percentile(distances, [25, 75])
    return distances <= q3 + iqr_factor * (q3 - q1)
