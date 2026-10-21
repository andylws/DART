# DART: Dual Prototypes for Test-Time OOD Detection Under Covariate Shift

Official implementation of **"Tracking the Discriminative Axis: Dual Prototypes for Test-Time OOD Detection Under Covariate Shift"** (WACV 2027).

DART (**D**iscriminative **A**xis **R**eal-time **T**racker) is a post-hoc, online OOD detector for test streams that mix in-distribution (ID) and OOD samples under the same, evolving covariate shift. For each feature layer it keeps an ID prototype and an OOD prototype. It updates them from unlabeled test batches and scores each sample by its relative distance to the two prototypes. DART needs only forward passes: there are no weight updates, no labels, and no access to training data.

## How DART works

For each layer, every test batch goes through these steps:

1. **Initialization** (first batch). Pseudo-label the batch with MSP + Otsu thresholding, ignoring feature outliers. The two group means become the initial prototypes.
2. **Scoring.** `RDS(x) = 1 - d(x, p_I) / (d(x, p_I) + d(x, p_O))` (Eq. 3). Higher means more likely ID.
3. **Prototype tracking.**
   - Otsu-threshold the RDS to get pseudo-labels.
   - Drop samples outside Tukey's fence in each group.
   - EMA-update the prototypes: `p ← α p + (1-α) p̂` (Eq. 4).
4. **Flip correction.** Every `k` batches, compare the prototypes with an MSP-based ID prototype and swap them if they have flipped (Eq. 5).
5. **Multi-layer fusion.** Average the per-layer scores (Sec. 3.4).

The method itself is self-contained in [`dart/dart.py`](dart/dart.py) and depends only on NumPy and PyTorch:

```python
from dart import MultiLayerDART

detector = MultiLayerDART(layers=["block1", "block2", "block3", "fc"], ema_alpha=0.8)
for x in test_stream:
    features_by_layer, logits = extract(model, x)   # dict layer -> (B, D), (B, C)
    scores = detector(features_by_layer, logits)    # (B,), higher = more likely ID
```

## Installation

```bash
conda create -n dart python=3.9 -y && conda activate dart
pip install -r requirements.txt
```

`tools/make_ood_c.py` also needs ImageMagick for the motion-blur and snow corruptions (`apt install libmagickwand-dev`).

## Running

```bash
# One OOD set, 15 corruptions (episodic: DART is reset for each corruption)
python main.py --dataset imagenet --arch regnet_y_16gf_swag_e2e --ood_dataset iNaturalist --corruption all

# Clean data
python main.py --dataset cifar100 --arch Hendrycks2020AugMix_WRN --ood_dataset svhn --corruption original

# Mixed covariate shift: one stream with interleaved corruptions
python main.py --dataset imagenet --arch resnet50 --ood_dataset iNaturalist \
    --corruption gaussian_noise snow --eval_type transition --shuffle
```

## Citation

```bibtex
@inproceedings{dart2027,
  title     = {Tracking the Discriminative Axis: Dual Prototypes for Test-Time OOD Detection Under Covariate Shift},
  author    = {TODO: add authors after camera-ready},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgements

- `third_party/glmcm_clip` is the CLIP implementation from [GL-MCM](https://github.com/AtsuMiyai/GL-MCM), based on [OpenAI CLIP](https://github.com/openai/CLIP) (MIT).
- `tools/make_ood_c.py` and the frost textures are adapted from [hendrycks/robustness](https://github.com/hendrycks/robustness) (Apache-2.0).
- The WRN-40-2 (AugMix) model comes from [RobustBench](https://github.com/RobustBench/robustbench).
