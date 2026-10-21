"""Frozen CLIP ViT-B/16 zero-shot classifier used as a DART backbone.

Uses the vendored GL-MCM copy of OpenAI CLIP (``third_party/glmcm_clip``).
"""

import os

import numpy as np
import torch
import torch.nn as nn

from third_party import glmcm_clip as clip

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_imagenet_classnames():
    """Return the 1000 cleaned ImageNet class names used by MCM/GL-MCM."""
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "imagenet_class_clean.npy")
    with open(path, "rb") as f:
        return list(np.load(f))


class CLIPWrapper(nn.Module):
    """CLIP with cached text features of the ID classes.

    The logits are cosine similarities to the class text embeddings (no logit
    scale), so the MSP reference inside DART is the MCM score.
    Inputs are expected in [0, 1].
    """

    def __init__(self, class_names, ckpt="ViT-B/16", text_template="a photo of a {}"):
        super().__init__()
        self.clip_model, _ = clip.load(ckpt, device="cuda")
        self.clip_model.float().eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        tokens = clip.tokenize([text_template.format(c) for c in class_names]).cuda()
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(tokens).float()
        self.register_buffer("text_features", text_feats / text_feats.norm(dim=-1, keepdim=True),
                             persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1).cuda(), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1).cuda(), persistent=False)

        # Each ResidualAttentionBlock returns (x, q, k, v) with x in (L, N, D);
        # x[0] is the CLS token after that block.
        resblocks = self.clip_model.visual.transformer.resblocks
        self.n_blocks = len(resblocks)
        self._block_outputs = {}
        for i, block in enumerate(resblocks):
            block.register_forward_hook(self._make_block_hook(i))

    def _make_block_hook(self, idx):
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            self._block_outputs[idx] = x[0].detach().float()
        return hook

    @torch.no_grad()
    def encode(self, x):
        """One visual forward.

        Returns:
            image_feat: (B, 512) L2-normalized post-projection embedding.
            block_feats: dict ``block index -> (B, 768)`` L2-normalized CLS
                features after ``ln_post``.
        """
        self._block_outputs = {}
        g, _ = self.clip_model.encode_image((x.cuda() - self.pixel_mean) / self.pixel_std)
        image_feat = g.float() / g.float().norm(dim=-1, keepdim=True)

        block_feats = {}
        for idx, cls in self._block_outputs.items():
            cls = self.clip_model.visual.ln_post(cls).float()
            block_feats[idx] = cls / cls.norm(dim=-1, keepdim=True)
        return image_feat, block_feats

    def logits(self, image_feat):
        return image_feat @ self.text_features.T

    def forward(self, x):
        return self.logits(self.encode(x)[0])
