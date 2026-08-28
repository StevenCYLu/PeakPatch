# PeakPatch

### What CLIP Knows but Cannot Say: Recovering Negation from Frozen Intermediate Features

Chen-Yi Lu, Yueh-Shao Chen, Somali Chaterji &mdash; Purdue University
**ECCV 2026**

**[Project page](https://stevencylu.github.io/PeakPatch/)** &nbsp;|&nbsp; **[arXiv:2607.23271](https://arxiv.org/abs/2607.23271)** &nbsp;|&nbsp; **[PDF](https://arxiv.org/pdf/2607.23271)**

---

> **Scope.** This repository covers the **negation MCQ** experiments: the released checkpoints, the
> evaluation, and the training pipeline behind them. Retrieval, text-to-image and LCD-analysis
> experiments are in the paper and on the project page.

> **Naming.** The paper calls the modules ECN and SCN; the code calls them `EmbeddingCorrector` and
> `ScoreCorrector`. They are the same modules.

## Results

NegBench MCQ accuracy (%), frozen CLIP ViT-B/32. Aff / Neg / Hyb are the affirmation, negation and
hybrid question templates; Avg is over all questions.

| | Aff | Neg | Hyb | **Avg** |
|---|:--:|:--:|:--:|:--:|
| CLIP, COCO | 69.1 | 6.8 | 39.3 | **39.3** |
| **PeakPatch, COCO** | **98.1** | **63.2** | **60.7** | **74.3** |
| CLIP, VOC2007 | 82.6 | 3.4 | 59.0 | **38.7** |
| **PeakPatch, VOC2007** | **99.7** | **57.9** | **62.2** | **65.5** |

Evaluation is deterministic: on one A100 the commands below reproduce 0.7433209 and 0.6547406
exactly.

> The CLIP rows above are this repository's own measurements. The baseline columns in the paper are
> quoted from the NegBench benchmark, so they differ from these by a few tenths of a point.

## Installation

```bash
uv sync
```

Requires Python &ge; 3.10 and a CUDA-capable GPU (the evaluation needs ~2 GB of VRAM). The CLIP
backbone is downloaded by `open_clip` on first use; set `CLIP_CACHE_DIR` to control where.

## Checkpoints

Shipped in `checkpoints/`, no download needed:

| File | Module | Params | Config |
|---|---|--:|---|
| `peakpatch_ecn.pt` | ECN | 4,728,833 | peak layer 8, anchor layer 6 |
| `peakpatch_scn.pt` | SCN | 493,825 | layers [7, 12], bound 0.2 |

## Data setup

The evaluation needs the NegBench MCQ CSVs plus the COCO and VOC images they reference.

1. **NegBench CSVs.** Download `COCO_val_mcq_llama3.1_rephrased.csv` and
   `VOC2007_mcq_llama3.1_rephrased.csv` from the
   [NegBench release](https://github.com/m1k2zoo/negbench) (see its `datasets.md`) into one
   directory; pass it as `--negbench-csv-dir`.

2. **COCO val2017.** Download from [cocodataset.org](https://cocodataset.org/#download). The CSV
   paths are relative, so lay the images out as `<coco-image-root>/data/coco/images/val2017/*.jpg`:

   ```bash
   mkdir -p coco_root/data/coco/images
   ln -s /path/to/val2017 coco_root/data/coco/images/val2017
   ```

3. **VOC2007 test.** Download
   [`VOCtest_06-Nov-2007.tar`](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
   and extract it so the images sit at
   `<voc-image-root>/data/voc2007/VOCdevkit/VOC2007/JPEGImages/*.jpg`:

   ```bash
   mkdir -p voc_root/data/voc2007 && tar xf VOCtest_06-Nov-2007.tar -C voc_root/data/voc2007
   ```

The three paths can also be given as the environment variables `NEGBENCH_CSV_DIR`,
`COCO_IMAGE_ROOT` and `VOC_IMAGE_ROOT`.

## Reproducing the paper's MCQ numbers

```bash
uv run python scripts/eval_mcq.py --tasks coco voc \
    --negbench-csv-dir /path/to/negbench_csvs \
    --coco-image-root /path/to/coco_root \
    --voc-image-root  /path/to/voc_root \
    --output results/peakpatch_mcq.json
```

```
coco_mcq-total_accuracy: 0.7433      voc_mcq-total_accuracy: 0.6547
coco_mcq-positive_accuracy: 0.9808   voc_mcq-positive_accuracy: 0.9971
coco_mcq-negative_accuracy: 0.6321   voc_mcq-negative_accuracy: 0.5790
coco_mcq-hybrid_accuracy: 0.6069     voc_mcq-hybrid_accuracy: 0.6221
```

The frozen-backbone baseline row:

```bash
uv run python scripts/eval_mcq.py --model clip --tasks coco voc \
    --negbench-csv-dir ... --coco-image-root ... --voc-image-root ...
```

Any fine-tuned ViT-B/32 checkpoint (NegCLIP, CoN-CLIP, ...) can be scored the same way with
`--model clip --clip-checkpoint /path/to/weights.pt`.

## Training

**1. Build the training data** from COCO train2014 &mdash; object-absence negated captions for the
ECN, and 4-way MCQ questions for the SCN. Both CSVs record absolute image paths:

```bash
uv run python scripts/prepare_coco_train.py \
    --coco-dir /path/to/coco --output-dir data/coco_train_negation
# -> coco_negcap_train2014.csv, coco_negmcq_train2014.csv
```

**2. Pre-extract frozen CLIP features.** Nothing in the backbone is trained, so its features are
computed once and reused. Pass `--model ViT-B-32` explicitly &mdash; the extractors default to
ViT-L/14:

```bash
# negcap features for the ECN, split 90/10
for split in train val; do
  uv run python scripts/extract_negcap_features.py \
      --input-csv data/coco_train_negation/coco_negcap_train2014.csv \
      --output-dir data/negcap/$split \
      --model ViT-B-32 --split $split --split-ratio 0.9 --seed 42
done

# per-layer [EOS] features for the SCN
uv run python scripts/extract_features.py \
    --input-csv data/coco_train_negation/coco_negmcq_train2014.csv \
    --image-root / --output-dir data/mcq/train \
    --model ViT-B-32 --layers 3 7 8 12
```

**3. Train.** The ECN and SCN are trained together from randomly initialised weights, in one run of
about an hour on a single A100:

```bash
uv run python scripts/train_joint.py --exp-name peakpatch \
    --negcap-train-dir data/negcap/train --negcap-val-dir data/negcap/val \
    --mcq-train-dir data/mcq/train --mcq-val-dir data/mcq/val \
    --target-layer 8 --anchor-layer 6 --sc-layers 7 12 \
    --epochs 10 --ec-batch-size 1024 --sc-batch-size 1024 \
    --lr 1e-4 --ec-lr-scale 0.1 --lambda-sc 1.0 --lambda-reg 0.1 \
    --output-dir results/joint
```

`scripts/train_ec.py` and `scripts/train_sc.py` train either module on its own, for ablations.

## Citation

```bibtex
@inproceedings{lu2026peakpatch,
  title         = {What CLIP Knows but Cannot Say: Recovering Negation
                   from Frozen Intermediate Features},
  author        = {Lu, Chen-Yi and Chen, Yueh-Shao and Chaterji, Somali},
  booktitle     = {European Conference on Computer Vision (ECCV)},
  year          = {2026},
  eprint        = {2607.23271},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

MIT &mdash; see [LICENSE](LICENSE).
