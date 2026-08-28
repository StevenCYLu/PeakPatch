"""Feature extraction utilities for CLIP intermediate layers.

This module provides functions to extract EOS token features from
CLIP's transformer layers, which are then saved for efficient training.

Supports optional EC (EmbeddingCorrector) chaining: when --ec-checkpoint is
provided, the frozen EC corrects the L12 text embedding before saving,
so SC can train on EC-corrected features.

Usage:
    # Standard extraction
    uv run python scripts/extract_features.py \
        --input-csv /path/to/data.csv \
        --image-root /path/to/images \
        --output-dir /path/to/output \
        --layers 6 8 12

    # EC-chained extraction (corrected L12)
    uv run python scripts/extract_features.py \
        --input-csv /path/to/data.csv \
        --image-root /path/to/images \
        --output-dir /path/to/output \
        --layers 6 8 12 \
        --ec-checkpoint results/embedding_corrector/layers_t8_a6/checkpoints/best_model.pt
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Model cache directory (configurable via env var)
CLIP_CACHE_DIR = os.environ.get("CLIP_CACHE_DIR") or None


def extract_layer_features(
    texts: List[str],
    model,
    tokenizer,
    layers: List[int] = None,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """Extract intermediate layer features from CLIP text encoder.

    Supports both standard CLIP and CustomTextCLIP (SigLIP) architectures
    via clip_utils abstractions.

    Args:
        texts: List of text strings
        model: CLIP model
        tokenizer: CLIP tokenizer
        layers: Which layers to extract (1-indexed)
        device: Device for computation

    Returns:
        Dict mapping layer_idx -> features [B, D]
    """
    from peakpatch.clip_utils import (
        _get_text_encoder, _apply_text_projection, get_eos_positions,
    )

    if layers is None:
        layers = [3, 8, 12]

    tokens = tokenizer(texts).to(device)

    text_enc = _get_text_encoder(model)
    cast_dtype = text_enc.transformer.get_cast_dtype()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        eos_pos = get_eos_positions(tokens, model)

        x = text_enc.token_embedding(tokens).to(cast_dtype)
        x = x + text_enc.positional_embedding[: x.size(1)].to(cast_dtype)
        attn_mask = text_enc.attn_mask if hasattr(text_enc, "attn_mask") else None

        batch_size = len(texts)
        batch_idx = torch.arange(batch_size, device=device)

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


def extract_layer_features_ec_corrected(
    texts: List[str],
    model,
    tokenizer,
    ec_model,
    layers: List[int],
    ec_target_layer: int = 8,
    ec_anchor_layer: int = 6,
    ec_alpha: float = None,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """Extract features with EC-corrected L12 in a single forward pass.

    SC layers (e.g. 6, 8) get standard CLS extraction (EOS -> ln_final ->
    text_projection -> normalize). EC anchor/target layers also get full
    token sequences (ln_final only, no projection) for the EC model.
    The final layer CLS feeds into EC, which produces the corrected output.

    Supports both standard CLIP and CustomTextCLIP (SigLIP) architectures.

    Args:
        texts: List of text strings
        model: CLIP model
        tokenizer: CLIP tokenizer
        ec_model: Frozen EmbeddingCorrector
        layers: SC layers to extract (e.g. [6, 8, 12])
        ec_target_layer: EC target layer (default 8)
        ec_anchor_layer: EC anchor layer (default 6)
        ec_alpha: Override EC alpha (None = use learned)
        device: Device for computation

    Returns:
        Dict mapping layer_idx -> features [B, D], with the final layer
        replaced by EC-corrected embedding.
    """
    from peakpatch.clip_utils import (
        _get_text_encoder, _apply_text_projection, get_eos_positions,
        compute_padding_mask,
    )

    tokens = tokenizer(texts).to(device)

    text_enc = _get_text_encoder(model)
    cast_dtype = text_enc.transformer.get_cast_dtype()
    final_layer = len(text_enc.transformer.resblocks)

    ec_layers = {ec_anchor_layer, ec_target_layer}
    all_needed = set(layers) | ec_layers | {final_layer}
    max_layer = max(all_needed)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        eos_pos = get_eos_positions(tokens, model)
        x = text_enc.token_embedding(tokens).to(cast_dtype)
        x = x + text_enc.positional_embedding[: x.size(1)].to(cast_dtype)
        attn_mask = text_enc.attn_mask if hasattr(text_enc, "attn_mask") else None

        B = len(texts)
        batch_idx = torch.arange(B, device=device)

        cls_features = {}
        token_seqs = {}

        for layer_idx, block in enumerate(text_enc.transformer.resblocks):
            x = block(x, attn_mask=attn_mask)
            current_layer = layer_idx + 1

            if current_layer not in all_needed:
                if current_layer >= max_layer:
                    break
                continue

            # CLS extraction for SC layers and final layer
            if current_layer in layers or current_layer == final_layer:
                eos_features = x[batch_idx, eos_pos]
                eos_normed = text_enc.ln_final(eos_features)
                eos_projected = _apply_text_projection(text_enc, eos_normed)
                cls_features[current_layer] = F.normalize(eos_projected.float(), dim=-1)

            # Full token sequences for EC anchor/target
            if current_layer in ec_layers:
                token_seqs[current_layer] = text_enc.ln_final(x).float()

            if current_layer >= max_layer:
                break

        # EC correction
        padding_mask = compute_padding_mask(tokens, model)
        text_cls = cls_features[final_layer]
        H_anchor = token_seqs[ec_anchor_layer]
        H_target = token_seqs[ec_target_layer]

        corrected_final, _ = ec_model.correct_embedding(
            text_cls, H_anchor, H_target, padding_mask, alpha=ec_alpha)

    # Build result: SC layers with EC-corrected final layer
    features = {}
    for layer in layers:
        if layer == final_layer:
            features[layer] = corrected_final
        else:
            features[layer] = cls_features[layer]
    return features


SHARD_SIZE = 200  # Save a shard every N batches


def _save_mcq_shard(shard_dir, shard_idx, buf, layers, save_token_ids):
    """Save a shard of MCQ extraction results."""
    data = {
        "image_emb": torch.cat(buf["image_emb"], dim=0),
        "mcq_labels": torch.tensor(buf["mcq_labels"]),
    }
    for layer in layers:
        data[f"text_layer_{layer}"] = torch.cat(buf["text_features"][layer], dim=0)
        data[f"mcq_layer_{layer}"] = torch.cat(buf["mcq_features"][layer], dim=0)
    if save_token_ids and buf["mcq_token_ids"]:
        data["mcq_token_ids"] = torch.cat(buf["mcq_token_ids"], dim=0)
    torch.save(data, shard_dir / f"shard_{shard_idx:04d}.pt")


def _concat_mcq_shards(shard_dir, output_dir, layers, save_token_ids, use_ec,
                        ec_target_layer=None, ec_anchor_layer=None, csv_path=""):
    """Concatenate all MCQ shards into final output files."""
    shard_files = sorted(shard_dir.glob("shard_*.pt"))
    n_shards = len(shard_files)
    print(f"Concatenating {n_shards} shards...")

    all_data = [torch.load(f, weights_only=True) for f in shard_files]

    image_tensor = torch.cat([d["image_emb"] for d in all_data])
    torch.save(image_tensor, output_dir / "image_emb.pt")
    print(f"  image_emb.pt: {image_tensor.shape}")

    for layer in layers:
        text_tensor = torch.cat([d[f"text_layer_{layer}"] for d in all_data])
        torch.save(text_tensor, output_dir / f"text_layer_{layer:02d}.pt")
        print(f"  text_layer_{layer:02d}.pt: {text_tensor.shape}")

        mcq_tensor = torch.cat([d[f"mcq_layer_{layer}"] for d in all_data])
        torch.save(mcq_tensor, output_dir / f"mcq_layer_{layer:02d}.pt")
        print(f"  mcq_layer_{layer:02d}.pt: {mcq_tensor.shape}")

    labels_tensor = torch.cat([d["mcq_labels"] for d in all_data])
    torch.save(labels_tensor, output_dir / "mcq_labels.pt")
    print(f"  mcq_labels.pt: {labels_tensor.shape}")

    if save_token_ids:
        token_ids_tensor = torch.cat([d["mcq_token_ids"] for d in all_data])
        torch.save(token_ids_tensor, output_dir / "mcq_token_ids.pt")
        print(f"  mcq_token_ids.pt: {token_ids_tensor.shape}")

    metadata = {
        "num_samples": len(image_tensor),
        "selected_layers": layers,
        "embed_dim": image_tensor.shape[-1],
        "source_csv": str(csv_path),
        "ec_corrected": use_ec,
    }
    if use_ec:
        metadata["ec_target_layer"] = ec_target_layer
        metadata["ec_anchor_layer"] = ec_anchor_layer
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json: {len(image_tensor)} samples")


def extract_dataset_features(
    csv_path: str,
    image_root: str,
    output_dir: str,
    model,
    tokenizer,
    preprocess,
    layers: List[int] = None,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    ec_model=None,
    ec_config: Optional[Dict] = None,
    save_token_ids: bool = False,
):
    """Extract and save features for an entire dataset.

    Uses shard-based checkpointing so partial progress survives timeouts.
    On restart, existing shards are skipped automatically.

    Args:
        csv_path: Path to CSV with columns:
            - image_path: relative path to image
            - caption_0, caption_1, caption_2, caption_3: MCQ options
            - correct_answer: correct option index (0-3)
        image_root: Root directory for images
        output_dir: Directory to save extracted features
        model: CLIP model
        tokenizer: CLIP tokenizer
        preprocess: Image preprocessing transform
        layers: Which layers to extract
        device: Device for computation
        batch_size: Batch size for extraction
        num_workers: Number of data loading workers
        ec_model: Optional frozen EmbeddingCorrector for L12 correction
        ec_config: EC configuration dict with target_layer, anchor_layer, ec_alpha
        save_token_ids: If True, tokenize MCQ options and save as mcq_token_ids.pt
    """
    if layers is None:
        layers = [3, 8, 12]

    use_ec = ec_model is not None
    ec_target_layer = ec_anchor_layer = ec_alpha = None
    if use_ec:
        ec_target_layer = ec_config.get("target_layer", 8)
        ec_anchor_layer = ec_config.get("anchor_layer", 6)
        ec_alpha = ec_config.get("ec_alpha", None)
        print(f"EC correction enabled: target={ec_target_layer}, anchor={ec_anchor_layer}")

    def _extract_fn(texts):
        if use_ec:
            return extract_layer_features_ec_corrected(
                texts=texts, model=model, tokenizer=tokenizer,
                ec_model=ec_model, layers=layers,
                ec_target_layer=ec_target_layer,
                ec_anchor_layer=ec_anchor_layer,
                ec_alpha=ec_alpha, device=device,
            )
        return extract_layer_features(
            texts=texts, model=model, tokenizer=tokenizer,
            layers=layers, device=device,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    if (output_dir / "metadata.json").exists():
        print(f"Already complete: {output_dir / 'metadata.json'} exists. Skipping.")
        return

    # Shard directory for checkpointing
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} samples from {csv_path}")

    # Find existing shards to resume from
    existing_shards = sorted(shard_dir.glob("shard_*.pt"))
    start_shard = len(existing_shards)
    start_batch = start_shard * SHARD_SIZE
    if start_shard > 0:
        print(f"  Resuming: found {start_shard} shards, skipping first {start_batch} batches")

    # Initialize shard buffer
    def _new_buf():
        return {
            "image_emb": [],
            "text_features": {layer: [] for layer in layers},
            "mcq_features": {layer: [] for layer in layers},
            "mcq_labels": [],
            "mcq_token_ids": [] if save_token_ids else None,
        }

    buf = _new_buf()
    shard_idx = start_shard
    batch_in_shard = 0

    # Process in batches with prefetching: load next batch on CPU while GPU processes current
    def _load_image(args):
        idx, row = args
        image_path = Path(image_root) / row["image_path"]
        try:
            img = Image.open(image_path).convert("RGB")
            return idx, preprocess(img)
        except Exception:
            return idx, None

    def _load_batch(batch_df):
        """Load images for a batch, return (images, valid_indices)."""
        items = list(batch_df.iterrows())
        results = list(io_pool.map(_load_image, items))
        images, valid_indices = [], []
        for idx, tensor in results:
            if tensor is not None:
                images.append(tensor)
                valid_indices.append(idx)
        return images, valid_indices

    batch_starts = list(range(0, len(df), batch_size))
    active_batches = [(i, s) for i, s in enumerate(batch_starts) if i >= start_batch]

    io_pool = ThreadPoolExecutor(max_workers=num_workers)
    prefetch_pool = ThreadPoolExecutor(max_workers=1)

    # Prefetch first batch
    if active_batches:
        _, first_start = active_batches[0]
        first_end = min(first_start + batch_size, len(df))
        prefetch_future = prefetch_pool.submit(_load_batch, df.iloc[first_start:first_end])

    for idx, (batch_num, start_idx) in enumerate(tqdm(active_batches, desc="Extracting features")):
        # Get prefetched result
        images, valid_indices = prefetch_future.result()

        # Kick off prefetch for NEXT batch before doing GPU work
        if idx + 1 < len(active_batches):
            _, next_start = active_batches[idx + 1]
            next_end = min(next_start + batch_size, len(df))
            prefetch_future = prefetch_pool.submit(_load_batch, df.iloc[next_start:next_end])

        if not images:
            batch_in_shard += 1
            if batch_in_shard >= SHARD_SIZE and buf["image_emb"]:
                _save_mcq_shard(shard_dir, shard_idx, buf, layers, save_token_ids)
                print(f"  Saved shard {shard_idx}")
                shard_idx += 1
                buf = _new_buf()
                batch_in_shard = 0
            continue

        images = torch.stack(images).to(device)

        # Extract image embeddings
        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_emb = model.encode_image(images, normalize=True)
        buf["image_emb"].append(image_emb.cpu())

        # Extract text features for correct captions
        correct_captions = []
        for vi in valid_indices:
            row = df.loc[vi]
            correct_idx = int(row["correct_answer"])
            correct_captions.append(row[f"caption_{correct_idx}"])

        text_features = _extract_fn(correct_captions)

        for layer in layers:
            buf["text_features"][layer].append(text_features[layer].cpu())

        # Extract MCQ features (all 4 options) -- batched
        all_options = []
        for vi in valid_indices:
            row = df.loc[vi]
            for i in range(4):
                all_options.append(row[f"caption_{i}"])
        mcq_features = _extract_fn(all_options)
        n_valid = len(valid_indices)
        for layer in layers:
            buf["mcq_features"][layer].append(
                mcq_features[layer].cpu().reshape(n_valid, 4, -1))

        # MCQ labels
        for vi in valid_indices:
            row = df.loc[vi]
            buf["mcq_labels"].append(int(row["correct_answer"]))

        # MCQ token IDs
        if save_token_ids:
            mcq_captions = []
            for vi in valid_indices:
                row = df.loc[vi]
                for i in range(4):
                    mcq_captions.append(row[f"caption_{i}"])
            mcq_tokens = tokenizer(mcq_captions)
            mcq_tokens = mcq_tokens.reshape(n_valid, 4, -1)
            buf["mcq_token_ids"].append(mcq_tokens)

        batch_in_shard += 1
        if batch_in_shard >= SHARD_SIZE:
            _save_mcq_shard(shard_dir, shard_idx, buf, layers, save_token_ids)
            print(f"  Saved shard {shard_idx}")
            shard_idx += 1
            buf = _new_buf()
            batch_in_shard = 0

    io_pool.shutdown()
    prefetch_pool.shutdown()

    # Save remaining buffer as final shard
    if buf["image_emb"]:
        _save_mcq_shard(shard_dir, shard_idx, buf, layers, save_token_ids)
        print(f"  Saved shard {shard_idx}")
        shard_idx += 1

    # Concatenate all shards into final files
    print("\nConcatenating shards into final files...")
    _concat_mcq_shards(
        shard_dir, output_dir, layers, save_token_ids, use_ec,
        ec_target_layer, ec_anchor_layer, csv_path)

    print(f"\nFeatures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP layer features for NegBench training"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input CSV with MCQ data",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory for images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="CLIP model name",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[3, 8, 12],
        help="Layers to extract features from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of threads for image loading",
    )
    parser.add_argument(
        "--save-token-ids",
        action="store_true",
        help="Save tokenized MCQ options as mcq_token_ids.pt [N, 4, 77]",
    )
    parser.add_argument(
        "--ec-checkpoint",
        type=str,
        default=None,
        help="Path to frozen EmbeddingCorrector checkpoint for L12 correction",
    )
    parser.add_argument(
        "--ec-target-layer",
        type=int,
        default=8,
        help="EC target layer (default: 8)",
    )
    parser.add_argument(
        "--ec-anchor-layer",
        type=int,
        default=6,
        help="EC anchor layer (default: 6)",
    )
    parser.add_argument(
        "--ec-alpha",
        type=float,
        default=None,
        help="Override EC alpha (default: use learned value)",
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load CLIP model directly via open_clip
    import open_clip

    print(f"Loading CLIP model: {args.model} ({args.pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained,
        cache_dir=CLIP_CACHE_DIR,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    # Load EC model if checkpoint provided
    ec_model = None
    ec_config = None
    if args.ec_checkpoint:
        from peakpatch.model import EmbeddingCorrector

        print(f"Loading EmbeddingCorrector from {args.ec_checkpoint}")
        ec_model = EmbeddingCorrector.from_checkpoint(args.ec_checkpoint, device=device)
        for p in ec_model.parameters():
            p.requires_grad = False
        ec_config = {
            "target_layer": args.ec_target_layer,
            "anchor_layer": args.ec_anchor_layer,
            "ec_alpha": args.ec_alpha,
        }
        learned_alpha = torch.exp(ec_model.log_alpha).item()
        effective = args.ec_alpha if args.ec_alpha is not None else learned_alpha
        print(f"  Target layer: {args.ec_target_layer}, Anchor layer: {args.ec_anchor_layer}")
        print(f"  Learned alpha: {learned_alpha:.4f}, effective: {effective:.4f}")

    # Extract features
    extract_dataset_features(
        csv_path=args.input_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        layers=args.layers,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ec_model=ec_model,
        ec_config=ec_config,
        save_token_ids=args.save_token_ids,
    )


if __name__ == "__main__":
    main()
