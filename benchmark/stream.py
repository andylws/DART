"""Build the ID/OOD test streams for the evaluation protocols in the paper.

* episodic: one stream per corruption; DART is reset between streams (Tab. 1-3).
* transition: one stream over all given corruptions / OOD sets, never reset.
  With ``shuffle`` the sources are interleaved (mixed OOD / mixed covariate
  shift, Tab. 4-5); without it they arrive in sequential blocks (continual
  OOD / continual covariate shift, Fig. 6).
"""

import logging

import torch

from .data import CORRUPTIONS, load_id, load_ood

logger = logging.getLogger(__name__)


def parse_corruptions(corruptions):
    if len(corruptions) == 1 and corruptions[0].lower() == "all":
        return list(CORRUPTIONS)
    return list(corruptions)


def _shuffled(x):
    return x[torch.randperm(x.size(0))]


def _cycle(items, length):
    return [items[i % len(items)] for i in range(length)]


def episodic_stream(corruption, args):
    """ID and OOD samples of a single corruption, truncated to equal size."""
    x_ind, _ = load_id(args.dataset, args.num_ex, corruption, args.severity, args.data_dir)
    n_per_ood = args.num_ex // len(args.ood_dataset)
    x_ood = torch.cat([load_ood(args.dataset, name, n_per_ood, corruption, args.severity, args.data_dir)[0]
                       for name in args.ood_dataset])
    if len(args.ood_dataset) > 1 and args.shuffle:
        x_ood = _shuffled(x_ood)
    return x_ind, x_ood[:x_ind.size(0)]


def transition_stream(corruptions, args):
    """One stream concatenating every corruption (ID) and corruption/OOD pair (OOD)."""
    n_per_corruption = args.num_ex // len(corruptions)
    x_ind = torch.cat([load_id(args.dataset, n_per_corruption, c, args.severity, args.data_dir)[0]
                       for c in corruptions])

    # Pair corruptions with OOD sets, cycling the shorter list.
    n_pairs = max(len(corruptions), len(args.ood_dataset))
    n_per_pair = args.num_ex // n_pairs
    pairs = zip(_cycle(corruptions, n_pairs), _cycle(args.ood_dataset, n_pairs))
    x_ood = torch.cat([load_ood(args.dataset, name, n_per_pair, c, args.severity, args.data_dir)[0]
                       for c, name in pairs])

    if args.shuffle:
        x_ind, x_ood = _shuffled(x_ind), _shuffled(x_ood)
    return x_ind, x_ood


def build_streams(args):
    """Yield ``(name, load_fn)`` per stream; ``load_fn()`` returns ``(x_ind, x_ood)``."""
    corruptions = parse_corruptions(args.corruption)
    if args.eval_type == "episodic" or (len(corruptions) == 1 and len(args.ood_dataset) == 1):
        for corruption in corruptions:
            yield corruption, lambda c=corruption: episodic_stream(c, args)
    else:
        yield "mixed" if args.shuffle else "continual", lambda: transition_stream(corruptions, args)
