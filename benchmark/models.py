"""Backbones used in the paper and the layers DART reads from each of them."""

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from .clip_wrapper import CLIPWrapper, get_imagenet_classnames

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ARCHS = {
    "cifar100": ["Hendrycks2020AugMix_WRN", "vit-tiny", "swin-tiny"],
    "imagenet": ["regnet_y_16gf_swag_e2e", "resnet50", "clip-vitb16"],
}

_VIT_TINY_LAYERS = [f"vit.encoder.layer.{i}" for i in range(12)] + ["classifier"]
_SWIN_TINY_LAYERS = [f"swin.encoder.layers.{stage}.blocks.{block}"
                     for stage, depth in enumerate([2, 2, 6, 2]) for block in range(depth)] + ["classifier"]
_REGNET_LAYERS = (["stem"]
                  + [f"trunk_output.block1.block1-{i}" for i in range(2)]
                  + [f"trunk_output.block2.block2-{i}" for i in range(4)]
                  + [f"trunk_output.block3.block3-{i}" for i in range(11)]
                  + ["trunk_output.block4.block4-0", "fc"])

# arch -> (single-layer setting, multi-layer setting)
LAYERS = {
    "Hendrycks2020AugMix_WRN": (["block3"], ["block1", "block2", "block3", "fc"]),
    "vit-tiny": (["vit.encoder.layer.11"], _VIT_TINY_LAYERS),
    "swin-tiny": (["swin.encoder.layers.3.blocks.1"], _SWIN_TINY_LAYERS),
    "resnet50": (["avgpool"], ["layer1", "layer2", "layer3", "layer4", "fc"]),
    "regnet_y_16gf_swag_e2e": (["trunk_output"], _REGNET_LAYERS),
    "clip-vitb16": (["clip_image"], ["clip_image_block_3", "clip_image_block_9", "clip_image"]),
}


@dataclass
class Backbone:
    model: nn.Module
    layers: List[str]
    input_size: Optional[int] = None  # resize inputs to this size before the forward pass


def _load_hf_cifar100(arch, ckpt_dir):
    from transformers import SwinForImageClassification, ViTForImageClassification

    if arch == "vit-tiny":
        model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.classifier = nn.Linear(192, 100)
        ckpt = "checkpoint_cifar100_vit_tiny.pt"
    else:
        model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        model.classifier = nn.Linear(768, 100)
        ckpt = "checkpoint_cifar100_swin_tiny.pt"
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt), map_location="cpu"))
    return model


def build_backbone(arch, dataset, ckpt_dir="./ckpt", multi_layer=True):
    if arch not in ARCHS[dataset]:
        raise ValueError(f"{arch} is not supported for {dataset}; choose from {ARCHS[dataset]}")

    input_size = None
    if arch == "Hendrycks2020AugMix_WRN":
        model = load_model(arch, ckpt_dir, dataset, ThreatModel.corruptions)
    elif arch in ("vit-tiny", "swin-tiny"):
        model = _load_hf_cifar100(arch, ckpt_dir)
        input_size = 224
    elif arch == "resnet50":
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        model = normalize_model(model, IMAGENET_MEAN, IMAGENET_STD)
    elif arch == "regnet_y_16gf_swag_e2e":
        model = tv_models.regnet_y_16gf(weights=tv_models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)
        model = normalize_model(model, IMAGENET_MEAN, IMAGENET_STD)
    else:  # clip-vitb16
        model = CLIPWrapper(get_imagenet_classnames(), ckpt="ViT-B/16")

    single, multi = LAYERS[arch]
    return Backbone(model=model.cuda().eval(), layers=multi if multi_layer else single,
                    input_size=input_size)
