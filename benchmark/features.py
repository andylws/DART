"""Multi-layer feature extraction with forward hooks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_wrapper import CLIPWrapper


def _base_module(model):
    """Strip the input-normalization wrapper added by ``normalize_model``."""
    if isinstance(model, nn.Sequential) and hasattr(model, "model"):
        return model.model
    return model


def _pool(feat):
    """Reduce a layer output to a (B, D) vector."""
    if feat.dim() == 4:  # CNN: (B, C, H, W) -> global average pooling
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)
    if feat.dim() == 3:  # Transformer: (B, L, D) -> first token
        return feat[:, 0, :]
    return feat


@torch.no_grad()
def extract_features(model, x, layers):
    """Run one forward pass and collect pooled features of ``layers``.

    Returns:
        features: dict ``layer -> (B, D)`` CPU tensor.
        logits: (B, C) CPU tensor.
    """
    x = x.cuda()
    if isinstance(model, CLIPWrapper):
        return _extract_clip(model, x, layers)

    base = _base_module(model)
    activations = {}

    def make_hook(name):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            activations[name] = out.detach()
        return hook

    handles = [base.get_submodule(layer).register_forward_hook(make_hook(layer)) for layer in layers]
    try:
        logits = model(x)
    finally:
        for handle in handles:
            handle.remove()

    if hasattr(logits, "logits"):  # Hugging Face model output
        logits = logits.logits
    features = {layer: _pool(activations[layer]).cpu() for layer in layers}
    return features, logits.cpu()


def _extract_clip(model, x, layers):
    image_feat, block_feats = model.encode(x)
    features = {}
    for layer in layers:
        if layer == "clip_image":
            features[layer] = image_feat.cpu()
        elif layer.startswith("clip_image_block_"):
            features[layer] = block_feats[int(layer.rsplit("_", 1)[-1])].cpu()
        else:
            raise ValueError(f"Unknown CLIP layer: {layer}")
    return features, model.logits(image_feat).cpu()
