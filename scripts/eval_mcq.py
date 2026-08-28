"""Reproduce the PeakPatch MCQ results on NegBench (COCO and VOC2007).

PeakPatch runs on a frozen CLIP ViT-B/32 backbone. The ECN rewrites the final
text embedding from intermediate-layer token states; the SCN then scores the
four options using per-layer [EOS] features, with the ECN-corrected embedding
substituted for the last layer ("chained" scoring, exactly as in the paper).

Reproducing Table 1 (the shipped checkpoints):

    python scripts/eval_mcq.py --tasks coco voc \
        --negbench-csv-dir /path/to/NegBench/evaluation_data/images \
        --coco-image-root /path/to/coco_root \
        --voc-image-root /path/to/voc_root

    -> COCO 74.33, VOC 65.47

The frozen-CLIP baseline row:

    python scripts/eval_mcq.py --model clip --tasks coco voc ...

    -> COCO 39.30, VOC 38.72

Image roots are the directory each CSV's relative ``image_path`` resolves
against: ``<coco-image-root>/data/coco/images/val2017/...`` and
``<voc-image-root>/data/voc2007/VOCdevkit/VOC2007/JPEGImages/...``.
See README.md for how to lay these out.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# open_clip download cache. None lets open_clip use its own default location.
CLIP_CACHE_DIR = os.environ.get("CLIP_CACHE_DIR") or None

# JPEG decode dominates wall-clock and leaves the GPU idle between batches, so
# each batch is decoded across a thread pool. Set --num-workers 1 to serialise.
IMAGE_WORKERS = 8
_IMAGE_POOL = None

CSV_NAMES = {
    "coco": "COCO_val_mcq_llama3.1_rephrased.csv",
    "voc": "VOC2007_mcq_llama3.1_rephrased.csv",
}

DEFAULT_ECN = REPO_ROOT / "checkpoints" / "peakpatch_ecn.pt"
DEFAULT_SCN = REPO_ROOT / "checkpoints" / "peakpatch_scn.pt"


def load_clip_model(device, clip_arch=None, clip_pretrained=None, clip_checkpoint=None):
    """Load a CLIP backbone, optionally overriding it with fine-tuned weights.

    Args:
        device: torch device.
        clip_arch: open_clip architecture name (default ViT-B-32-quickgelu).
        clip_pretrained: pretrained tag (default openai).
        clip_checkpoint: optional path to a fine-tuned ViT-B/32 state dict
            (NegCLIP, CoN-CLIP, ...). Loaded non-strictly over the base model.

    Returns:
        (model, preprocess, tokenizer)
    """
    arch = clip_arch or "ViT-B-32-quickgelu"
    pretrained = clip_pretrained or "openai"
    print(f"  Architecture: {arch}, pretrained: {pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, cache_dir=CLIP_CACHE_DIR)

    if clip_checkpoint is not None:
        print(f"  Overriding weights from: {clip_checkpoint}")
        ckpt = torch.load(clip_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)

    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(arch)
    print(f"  Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, preprocess, tokenizer


def extract_sc_features(tokens, clip_model, layers):
    """Extract per-layer CLS-projected features for ScoreCorrector.

    For each requested layer, extracts the EOS token hidden state, applies
    ln_final and text_projection, then normalizes. This matches the
    pre-extraction pipeline in scripts/extract_features.py.

    Supports both standard CLIP and CustomTextCLIP (SigLIP).

    Args:
        tokens: [B, S] tokenized text (on device)
        clip_model: CLIP model
        layers: list of layer indices (1-indexed, e.g. [3, 8, 12])

    Returns:
        Dict[int, Tensor]: {layer_idx: [B, D]} normalized CLS-projected features
    """
    from peakpatch.clip_utils import (
        _get_text_encoder,
        _apply_text_projection,
        get_eos_positions,
    )

    text_enc = _get_text_encoder(clip_model)
    cast_dtype = text_enc.transformer.get_cast_dtype()
    device = tokens.device

    eos_pos = get_eos_positions(tokens, clip_model)
    x = text_enc.token_embedding(tokens).to(cast_dtype)
    x = x + text_enc.positional_embedding[:x.size(1)].to(cast_dtype)

    attn_mask = text_enc.attn_mask if hasattr(text_enc, "attn_mask") else None
    batch_idx = torch.arange(tokens.shape[0], device=device)

    features = {}
    for layer_idx, block in enumerate(text_enc.transformer.resblocks):
        x = block(x, attn_mask=attn_mask)
        current_layer = layer_idx + 1
        if current_layer in layers:
            eos_features = x[batch_idx, eos_pos]
            eos_normed = text_enc.ln_final(eos_features)
            eos_projected = _apply_text_projection(text_enc, eos_normed)
            features[current_layer] = F.normalize(eos_projected.float(), dim=-1)

    return features


def build_peakpatch_system(ec_checkpoint, sc_checkpoint, device):
    """Load the frozen CLIP backbone together with the ECN and SCN modules.

    The backbone architecture is read from the ECN checkpoint config, so the
    modules are always paired with the encoder they were trained against.

    Args:
        ec_checkpoint: path to the ECN weights.
        sc_checkpoint: path to the SCN weights.
        device: torch device.

    Returns:
        (system, preprocess, tokenizer), where system is the tuple consumed by
        ``eval_mcq``: (clip_model, sc_model, ec_model, ec_config, ec_alpha, sc_layers).
    """
    from peakpatch.model import EmbeddingCorrector, ScoreCorrector

    checkpoint = torch.load(ec_checkpoint, map_location="cpu", weights_only=False)
    ec_config = checkpoint.get("config", {})
    clip_arch = ec_config.get("clip_model", "ViT-B-32")
    clip_pretrained = ec_config.get("clip_pretrained", "openai")
    print(f"  Backbone: {clip_arch} ({clip_pretrained}), frozen")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_arch, pretrained=clip_pretrained, cache_dir=CLIP_CACHE_DIR)
    clip_model = clip_model.to(device).eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    tokenizer = open_clip.get_tokenizer(clip_arch)

    ec_model = EmbeddingCorrector.from_checkpoint(ec_checkpoint, device)
    ec_alpha = torch.exp(ec_model.log_alpha).item()
    print(f"  ECN: {ec_model.count_parameters():,} params, "
          f"peak layer {ec_config.get('target_layer', 12)}, "
          f"anchor layer {ec_config.get('anchor_layer', 10)}, alpha {ec_alpha:.3f}")

    sc_model = ScoreCorrector.from_checkpoint(sc_checkpoint, device)
    sc_layers = sc_model.selected_layers
    print(f"  SCN: {sum(p.numel() for p in sc_model.parameters()):,} params, "
          f"layers {sc_layers}, max_correction {sc_model.max_correction}")

    system = (clip_model, sc_model, ec_model, ec_config, ec_alpha, sc_layers)
    return system, preprocess, tokenizer


def _preprocess_one(args):
    path, preprocess = args
    with Image.open(path) as img:
        return preprocess(img.convert("RGB"))


def load_image_batch(paths, preprocess):
    """Decode and preprocess a batch of images, preserving input order.

    Uses ThreadPoolExecutor.map, which yields results in argument order, so the
    stacked batch is identical to decoding serially.
    """
    if IMAGE_WORKERS <= 1:
        return [_preprocess_one((p, preprocess)) for p in paths]

    global _IMAGE_POOL
    if _IMAGE_POOL is None:
        _IMAGE_POOL = ThreadPoolExecutor(max_workers=IMAGE_WORKERS)
    return list(_IMAGE_POOL.map(_preprocess_one, [(p, preprocess) for p in paths]))


def eval_mcq(model, preprocess, tokenizer, csv_path, image_root, device,
             peakpatch_system=None, batch_size=64):
    """Score a NegBench MCQ CSV and return accuracy broken down by answer type.

    Args:
        model: CLIP backbone, used directly when peakpatch_system is None.
        preprocess: CLIP image transform.
        tokenizer: CLIP tokenizer.
        csv_path: NegBench MCQ CSV (image_path, caption_0..3, correct_answer,
            correct_answer_template).
        image_root: directory the CSV's relative image_path entries resolve against.
        device: torch device.
        peakpatch_system: if given, the tuple built by ``build_peakpatch_system``;
            otherwise the frozen CLIP backbone is scored on its own.
        batch_size: questions per batch.

    Returns:
        dict of total/positive/negative/hybrid accuracy plus question counts.
    """
    print(f"\n--- MCQ Evaluation: {csv_path.name} ---")
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"  {n} questions")

    correct_by_type = {"positive": 0, "negative": 0, "hybrid": 0}
    total_by_type = {"positive": 0, "negative": 0, "hybrid": 0}
    total_correct = 0

    # Process in batches
    for start in tqdm(range(0, n, batch_size), desc="MCQ"):
        end = min(start + batch_size, n)
        batch_df = df.iloc[start:end]
        bs = len(batch_df)

        # Load images
        images = load_image_batch(
            [image_root / p for p in batch_df["image_path"]], preprocess)

        image_batch = torch.stack(images).to(device)

        # Get captions (4 per image) - collect all caption_0 first, then
        # all caption_1, etc. (matches NegBench ordering for correct reshape)
        all_captions = []
        for i in range(4):
            for _, row in batch_df.iterrows():
                all_captions.append(row[f"caption_{i}"])

        tokens = tokenizer(all_captions).to(device)

        with torch.no_grad():
            if peakpatch_system is not None:
                from peakpatch.clip_utils import (
                    compute_padding_mask,
                    extract_token_sequences,
                )

                (clip_model, sc_model, ec_model, ec_config,
                 ec_alpha, sc_layers) = peakpatch_system

                image_features = F.normalize(
                    clip_model.encode_image(image_batch).float(), dim=-1)

                # Per-layer [EOS] features the SCN scores from: [4*bs, D] -> [bs, 4, D].
                sc_feats = extract_sc_features(tokens, clip_model, sc_layers)
                sc_mcq = {
                    layer: sc_feats[layer].view(4, bs, -1).permute(1, 0, 2)
                    for layer in sc_layers
                }

                # ECN: rewrite the final text embedding from peak-layer token states.
                text_cls = clip_model.encode_text(tokens)
                text_cls = F.normalize(text_cls.float(), dim=-1)

                target_layer = ec_config.get("target_layer", 12)
                anchor_layer = ec_config.get("anchor_layer", 10)
                no_anchor = ec_model.no_anchor
                ec_layers = [target_layer] if no_anchor else [anchor_layer, target_layer]
                hidden = extract_token_sequences(tokens, clip_model, layers=ec_layers)
                H_anchor = hidden[target_layer] if no_anchor else hidden[anchor_layer]
                H_target = hidden[target_layer]
                padding_mask = compute_padding_mask(tokens, clip_model)
                ec_corrected, _ = ec_model.correct_embedding(
                    text_cls, H_anchor, H_target, padding_mask, alpha=ec_alpha)
                ec_corrected = ec_corrected.view(4, bs, -1).permute(1, 0, 2)

                # Chained scoring: the SCN reads the ECN-corrected embedding in
                # place of the backbone's final layer.
                sc_mcq[max(sc_layers)] = ec_corrected
                logits, _ = sc_model.score_mcq(sc_mcq, image_features)
            else:
                image_features = F.normalize(model.encode_image(image_batch), dim=-1)
                text_features = F.normalize(model.encode_text(tokens), dim=-1)
                # [4*bs, D] -> [4, bs, D], then [bs, D] x [4, bs, D] -> [bs, 4]
                text_features = text_features.view(4, bs, -1)
                logits = torch.einsum("bf,nbf->bn", image_features, text_features)

            predicted = torch.argmax(logits, dim=1).cpu()

        for i, (_, row) in enumerate(batch_df.iterrows()):
            correct_idx = int(row["correct_answer"])
            answer_type = row["correct_answer_template"]
            total_by_type[answer_type] += 1
            if predicted[i].item() == correct_idx:
                total_correct += 1
                correct_by_type[answer_type] += 1

    results = {
        "total_accuracy": total_correct / n,
        "positive_accuracy": correct_by_type["positive"] / max(total_by_type["positive"], 1),
        "negative_accuracy": correct_by_type["negative"] / max(total_by_type["negative"], 1),
        "hybrid_accuracy": correct_by_type["hybrid"] / max(total_by_type["hybrid"], 1),
        "n_questions": n,
        "n_by_type": total_by_type,
    }

    print(f"  Total accuracy: {results['total_accuracy']:.4f}")
    print(f"  Positive: {results['positive_accuracy']:.4f} ({total_by_type['positive']})")
    print(f"  Negative: {results['negative_accuracy']:.4f} ({total_by_type['negative']})")
    print(f"  Hybrid:   {results['hybrid_accuracy']:.4f} ({total_by_type['hybrid']})")
    return results


def main():
    global CLIP_CACHE_DIR, IMAGE_WORKERS

    parser = argparse.ArgumentParser(
        description="PeakPatch MCQ evaluation on NegBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=["peakpatch", "clip"], default="peakpatch",
                        help="peakpatch = frozen CLIP + ECN + SCN; clip = backbone only")
    parser.add_argument("--tasks", nargs="+", choices=["coco", "voc"], default=["coco", "voc"])

    parser.add_argument("--ec-checkpoint", type=Path, default=DEFAULT_ECN,
                        help="ECN weights")
    parser.add_argument("--sc-checkpoint", type=Path, default=DEFAULT_SCN,
                        help="SCN weights")
    parser.add_argument("--ec-alpha", type=float, default=None,
                        help="Override the ECN correction scale learned at training time")
    parser.add_argument("--sc-max-correction", type=float, default=None,
                        help="Override the SCN correction bound")

    parser.add_argument("--clip-arch", default=None, help="open_clip architecture (baseline only)")
    parser.add_argument("--clip-pretrained", default=None, help="pretrained tag (baseline only)")
    parser.add_argument("--clip-checkpoint", type=Path, default=None,
                        help="Fine-tuned ViT-B/32 weights to evaluate instead of OpenAI CLIP")
    parser.add_argument("--clip-cache-dir", default=None,
                        help="open_clip download cache (env: CLIP_CACHE_DIR)")

    parser.add_argument("--negbench-csv-dir", type=Path,
                        default=os.environ.get("NEGBENCH_CSV_DIR"),
                        help="Directory holding the NegBench MCQ CSVs (env: NEGBENCH_CSV_DIR)")
    parser.add_argument("--coco-image-root", type=Path,
                        default=os.environ.get("COCO_IMAGE_ROOT"),
                        help="Root that COCO image_path entries resolve against (env: COCO_IMAGE_ROOT)")
    parser.add_argument("--voc-image-root", type=Path,
                        default=os.environ.get("VOC_IMAGE_ROOT"),
                        help="Root that VOC image_path entries resolve against (env: VOC_IMAGE_ROOT)")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=IMAGE_WORKERS,
                        help="Threads used to decode images (1 = serial)")
    parser.add_argument("--output", type=Path, default=None, help="Write results as JSON")
    args = parser.parse_args()

    if args.clip_cache_dir:
        CLIP_CACHE_DIR = args.clip_cache_dir
    IMAGE_WORKERS = args.num_workers

    if args.negbench_csv_dir is None:
        parser.error("--negbench-csv-dir is required (or set NEGBENCH_CSV_DIR)")
    roots = {"coco": args.coco_image_root, "voc": args.voc_image_root}
    for task in args.tasks:
        if roots[task] is None:
            parser.error(f"--{task}-image-root is required for --tasks {task}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Tasks:  {args.tasks}")

    peakpatch_system = None
    if args.model == "peakpatch":
        for path, name in ((args.ec_checkpoint, "ECN"), (args.sc_checkpoint, "SCN")):
            if not Path(path).exists():
                parser.error(f"{name} checkpoint not found: {path}")
        peakpatch_system, preprocess, tokenizer = build_peakpatch_system(
            args.ec_checkpoint, args.sc_checkpoint, device)
        clip_model, sc_model, ec_model, ec_config, ec_alpha, sc_layers = peakpatch_system
        if args.ec_alpha is not None:
            ec_alpha = args.ec_alpha
            print(f"  ECN alpha overridden to: {ec_alpha}")
        if args.sc_max_correction is not None:
            sc_model.max_correction = args.sc_max_correction
            print(f"  SCN max_correction overridden to: {args.sc_max_correction}")
        peakpatch_system = (clip_model, sc_model, ec_model, ec_config,
                            ec_alpha, sc_layers)
        model = clip_model
    else:
        model, preprocess, tokenizer = load_clip_model(
            device, clip_arch=args.clip_arch, clip_pretrained=args.clip_pretrained,
            clip_checkpoint=args.clip_checkpoint)

    results = {"model": args.model}
    for task in args.tasks:
        csv_path = Path(args.negbench_csv_dir) / CSV_NAMES[task]
        if not csv_path.exists():
            print(f"\nSkipping {task}: CSV not found at {csv_path}")
            continue
        task_results = eval_mcq(
            model, preprocess, tokenizer, csv_path, Path(roots[task]), device,
            peakpatch_system=peakpatch_system, batch_size=args.batch_size)
        for key, value in task_results.items():
            results[f"{task}_mcq-{key}"] = value

    print("\n" + "=" * 60)
    print(f"RESULTS: {args.model}")
    print("=" * 60)
    for key, value in results.items():
        if key == "model":
            continue
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
