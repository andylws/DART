"""Evaluate DART on test-time OOD detection under covariate shift."""

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from benchmark.models import ARCHS, build_backbone
from benchmark.evaluate import evaluate_stream
from benchmark.stream import build_streams
from dart import MultiLayerDART

logger = logging.getLogger("dart")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Benchmark
    p.add_argument("--dataset", default="imagenet", choices=list(ARCHS))
    p.add_argument("--arch", default="regnet_y_16gf_swag_e2e",
                   choices=sorted({a for archs in ARCHS.values() for a in archs}))
    p.add_argument("--ood_dataset", nargs="+", default=["iNaturalist"],
                   help="OOD set(s). Several sets are mixed into one OOD stream.")
    p.add_argument("--corruption", nargs="+", default=["all"],
                   help="Corruption(s) applied to both ID and OOD; 'original' = clean, 'all' = 15 corruptions.")
    p.add_argument("--severity", type=int, default=5, choices=range(1, 6))
    p.add_argument("--eval_type", default="episodic", choices=["episodic", "transition"],
                   help="episodic: one stream per corruption, DART reset in between. "
                        "transition: a single stream over all corruptions/OOD sets.")
    p.add_argument("--shuffle", action="store_true",
                   help="transition only: interleave the sources (mixed) instead of sequential blocks (continual).")
    p.add_argument("--num_ex", type=int, default=10000, help="Number of ID (and OOD) test samples.")
    p.add_argument("--batch_size", type=int, default=100)
    # DART
    p.add_argument("--ema_alpha", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=5.0, help="Softmax temperature of the MSP reference.")
    p.add_argument("--iqr_factor", type=float, default=1.5, help="Tukey fence multiplier.")
    p.add_argument("--flip_weight", type=float, default=2.0, help="Distance ratio for flip detection (Eq. 5).")
    p.add_argument("--check_interval", type=int, default=10, help="Run the flip check every k batches.")
    p.add_argument("--no_flip", action="store_true", help="Disable flip correction (DART-NoFlip).")
    p.add_argument("--feature_layer", default="multi", choices=["single", "multi"])
    p.add_argument("--layer_index", type=int, default=None,
                   help="Use only this layer of the multi-layer list (single-layer ablation).")
    # Misc
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--ckpt_dir", default="./ckpt")
    p.add_argument("--save_dir", default="./results")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_logging(save_dir):
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(save_dir, "dart_log.txt"))):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def run_name(args):
    ood = "+".join(args.ood_dataset)
    corruption = args.corruption[0] if len(args.corruption) == 1 else "+".join(args.corruption)
    name = f"{args.arch}_{args.dataset}_{ood}_{corruption}_s{args.severity}_{args.eval_type}"
    if args.eval_type == "transition":
        name += "_mixed" if args.shuffle else "_continual"
    if args.no_flip:
        name += "_noflip"
    if args.layer_index is not None:
        name += f"_layer{args.layer_index}"
    return f"{name}_seed{args.seed}"


def main():
    args = parse_args()
    save_dir = os.path.join(args.save_dir, run_name(args))
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(save_dir)
    logger.info(f"Arguments: {vars(args)}")

    backbone = build_backbone(args.arch, args.dataset, args.ckpt_dir, multi_layer=args.feature_layer == "multi")
    if args.layer_index is not None:
        backbone.layers = [backbone.layers[args.layer_index]]
    logger.info(f"DART layers: {backbone.layers}")

    dart_kwargs = dict(ema_alpha=args.ema_alpha, temperature=args.temperature, iqr_factor=args.iqr_factor,
                       flip_weight=args.flip_weight, check_interval=args.check_interval, flip=not args.no_flip)

    results = {}
    for name, load_stream in build_streams(args):
        set_seed(args.seed)
        x_ind, x_ood = load_stream()
        logger.info(f"[{name}] {x_ind.size(0)} ID / {x_ood.size(0)} OOD samples")

        detector = MultiLayerDART(backbone.layers, **dart_kwargs)
        res = evaluate_stream(detector, backbone, x_ind, x_ood, args.batch_size)
        results[name] = res
        logger.info(f"[{name}] AUROC={res['auroc']:.4f} FPR95={res['fpr95']:.4f} | "
                    f"batch-avg AUROC={res['avg_auroc']:.4f} FPR95={res['avg_fpr95']:.4f}")

    keys = ["auroc", "fpr95", "avg_auroc", "avg_fpr95"]
    if len(results) > 1:
        results["mean"] = {k: float(np.mean([r[k] for r in results.values()])) for k in keys}
        logger.info("[mean] " + " ".join(f"{k}={results['mean'][k]:.4f}" for k in keys))

    with open(os.path.join(save_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stream"] + keys)
        for name, res in results.items():
            writer.writerow([name] + [f"{res[k]:.4f}" for k in keys])
    with open(os.path.join(save_dir, "batch_metrics.json"), "w") as f:
        json.dump({name: {k: res[k] for k in ("batch_aurocs", "batch_fprs")}
                   for name, res in results.items() if name != "mean"}, f)
    logger.info(f"Saved results to {save_dir}")


if __name__ == "__main__":
    main()
