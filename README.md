# PeakPatch

### What CLIP Knows but Cannot Say: Recovering Negation from Frozen Intermediate Features

Chen-Yi Lu, Yueh-Shao Chen, Somali Chaterji &mdash; Purdue University
**ECCV 2026**

**[Project page](https://stevencylu.github.io/PeakPatch/)** &nbsp;|&nbsp; **[arXiv:2607.23271](https://arxiv.org/abs/2607.23271)** &nbsp;|&nbsp; **[PDF](https://arxiv.org/pdf/2607.23271)**

---

## Results

NegBench MCQ accuracy (%), frozen CLIP ViT-B/32. Aff / Neg / Hyb are the affirmation, negation and
hybrid question templates; Avg is over all questions.

| | Aff | Neg | Hyb | **Avg** |
|---|:--:|:--:|:--:|:--:|
| CLIP, COCO | 69.1 | 6.8 | 39.3 | **39.3** |
| **PeakPatch, COCO** | **98.1** | **63.2** | **60.7** | **74.3** |
| CLIP, VOC2007 | 82.6 | 3.4 | 59.0 | **38.7** |
| **PeakPatch, VOC2007** | **99.7** | **57.9** | **62.2** | **65.5** |

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

## Evaluation

Score the released checkpoints on the NegBench MCQ sets:

```bash
uv run scripts/eval_mcq.py --tasks coco voc \
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

`--tasks` selects the sets to run, and `--output` writes the per-task accuracies as JSON. Use
`--model clip` to score the frozen backbone instead of PeakPatch:

```bash
uv run scripts/eval_mcq.py --model clip --tasks coco voc \
    --negbench-csv-dir ... --coco-image-root ... --voc-image-root ...
```

Any fine-tuned ViT-B/32 checkpoint (NegCLIP, CoN-CLIP, ...) can be scored the same way with
`--model clip --clip-checkpoint /path/to/weights.pt`.

## Training

The released checkpoints were trained on CC12M-derived negation data in the
[NegBench](https://github.com/m1k2zoo/negbench) CC12M-NegCap / CC12M-NegMCQ format:

| File | Used for | Rows | Correct-answer templates |
|---|---|--:|---|
| `cc12m_negcap_500k.csv.gz` | ECN (negated captions) | 1,176,307 | &mdash; |
| `cc12m_negmcq_500k.csv.gz` | SCN (4-way MCQ) | 391,823 | 130,766 hybrid, 130,590 negative, 110,706 positive, 19,761 modified positive |

Both files are attached to the
[`training-data` release](https://github.com/StevenCYLu/PeakPatch/releases/tag/training-data).
Image paths are relative (`images/000000001.jpg`), so pass the directory holding `images/` as
`--image-root`. The images themselves are not redistributed. Their IDs are keys from our own CC12M
download, not row numbers of the public `cc12m.tsv`.

```bash
mkdir -p data/cc12m && cd data/cc12m
gh release download training-data -R StevenCYLu/PeakPatch   # or download both files from the page
cd ../..
```

The MCQ set matters most. Its correct answers are often negated, as in NegBench. An SCN trained
only on questions whose correct answer is un-negated learns to reject every option that contains a
negation. It then scores below chance on NegBench (see the note at the end of this section).

**1. Split the MCQ CSV** 80/20 with seed 42:

```bash
uv run scripts/split_mcq_csv.py \
    --input-csv data/cc12m/cc12m_negmcq_500k.csv.gz --output-dir data/mcq_split
# -> train.csv (313,458 rows), val.csv (78,365 rows)
```

**2. Pre-extract frozen CLIP features.** Nothing in the backbone is trained, so its features are
computed once and reused. Pass `--model ViT-B-32` explicitly &mdash; the extractors default to
ViT-L/14:

```bash
# negcap features for the ECN, split 90/10
for split in train val; do
  uv run scripts/extract_negcap_features.py \
      --input-csv data/cc12m/cc12m_negcap_500k.csv.gz \
      --image-root /path/to/cc12m_root --output-dir data/negcap/$split \
      --model ViT-B-32 --split $split --split-ratio 0.9 --seed 42
done

# per-layer [EOS] features and token IDs for the SCN
for split in train val; do
  uv run scripts/extract_features.py \
      --input-csv data/mcq_split/$split.csv \
      --image-root /path/to/cc12m_root --output-dir data/mcq/$split \
      --model ViT-B-32 --layers 7 12 --save-token-ids
done
```

**3. Train.** The ECN and SCN are trained together end-to-end, in one run of about 2.7 hours on a
single A100:

```bash
uv run scripts/train_joint.py --exp-name peakpatch \
    --negcap-train-dir data/negcap/train --negcap-val-dir data/negcap/val \
    --mcq-train-dir data/mcq/train --mcq-val-dir data/mcq/val \
    --target-layer 8 --anchor-layer 6 --sc-layers 7 12 \
    --epochs 10 --ec-batch-size 1024 --sc-batch-size 1024 \
    --lr 1e-4 --ec-lr-scale 0.1 --lambda-sc 1.0 --lambda-reg 0.1 \
    --output-dir results/joint
```

Evaluate `results/joint/peakpatch/checkpoints/best_ec.pt` and `best_sc.pt` with
`scripts/eval_mcq.py --ec-checkpoint ... --sc-checkpoint ...`. On this data the best validation
SCN accuracy is around 0.82&ndash;0.87, well below what the COCO ablation below reaches. That is expected,
because these questions are harder.

`scripts/train_ec.py` and `scripts/train_sc.py` train either module on its own, for ablations.

> **COCO ablation.** `scripts/prepare_coco_train.py` builds negation data from COCO train2014. Every
> MCQ it writes has the un-negated caption as the correct answer. A model trained on it reaches
> ~96% validation accuracy but only ~23% on NegBench COCO MCQ, so do not use it to reproduce the
> results above.

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

Apache 2.0 &mdash; see [LICENSE](LICENSE).
