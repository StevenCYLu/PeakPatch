"""Extract negcap features for any CLIP model."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

CLIP_CACHE_DIR = os.environ.get("CLIP_CACHE_DIR") or None
SHARD_SIZE = 500


def extract_text_l12_cls(texts, model, tokenizer, device):
    """Extract normalized L12 CLS embeddings and token IDs for a batch of texts."""
    from peakpatch.clip_utils import (
        _get_text_encoder, _apply_text_projection, get_eos_positions,
    )

    tokens = tokenizer(texts).to(device)
    text_enc = _get_text_encoder(model)
    cast_dtype = text_enc.transformer.get_cast_dtype()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        eos_pos = get_eos_positions(tokens, model)
        x = text_enc.token_embedding(tokens).to(cast_dtype)
        x = x + text_enc.positional_embedding[:x.size(1)].to(cast_dtype)
        attn_mask = text_enc.attn_mask if hasattr(text_enc, "attn_mask") else None
        batch_idx = torch.arange(len(texts), device=device)

        for layer_idx, block in enumerate(text_enc.transformer.resblocks):
            x = block(x, attn_mask=attn_mask)

        eos_features = x[batch_idx, eos_pos]
        eos_normed = text_enc.ln_final(eos_features)
        eos_projected = _apply_text_projection(text_enc, eos_normed)
        cls_emb = F.normalize(eos_projected.float(), dim=-1)

    return cls_emb, tokens.cpu()


def save_shard(shard_dir, shard_idx, image_emb, orig_cls, neg_cls, orig_ids, neg_ids):
    """Save a shard of extracted features."""
    torch.save(torch.cat(image_emb, dim=0), shard_dir / f"image_emb_{shard_idx:04d}.pt")
    torch.save(torch.cat(orig_cls, dim=0), shard_dir / f"orig_cls_{shard_idx:04d}.pt")
    torch.save(torch.cat(neg_cls, dim=0), shard_dir / f"neg_cls_{shard_idx:04d}.pt")
    torch.save(torch.cat(orig_ids, dim=0), shard_dir / f"orig_ids_{shard_idx:04d}.pt")
    torch.save(torch.cat(neg_ids, dim=0), shard_dir / f"neg_ids_{shard_idx:04d}.pt")


def concat_shards(shard_dir, output_dir):
    """Concatenate all shards into final output files."""
    shard_files = sorted(shard_dir.glob("image_emb_*.pt"))
    n_shards = len(shard_files)
    print(f"Concatenating {n_shards} shards...")

    image_emb = torch.cat([torch.load(shard_dir / f"image_emb_{i:04d}.pt", weights_only=True) for i in range(n_shards)])
    orig_cls = torch.cat([torch.load(shard_dir / f"orig_cls_{i:04d}.pt", weights_only=True) for i in range(n_shards)])
    neg_cls = torch.cat([torch.load(shard_dir / f"neg_cls_{i:04d}.pt", weights_only=True) for i in range(n_shards)])
    orig_ids = torch.cat([torch.load(shard_dir / f"orig_ids_{i:04d}.pt", weights_only=True) for i in range(n_shards)])
    neg_ids = torch.cat([torch.load(shard_dir / f"neg_ids_{i:04d}.pt", weights_only=True) for i in range(n_shards)])

    torch.save(image_emb, output_dir / "image_emb.pt")
    torch.save(orig_cls, output_dir / "original_layer_12.pt")
    torch.save(neg_cls, output_dir / "negated_layer_12.pt")
    torch.save(orig_ids, output_dir / "original_token_ids.pt")
    torch.save(neg_ids, output_dir / "negated_token_ids.pt")

    print(f"  image_emb.pt: {image_emb.shape}")
    print(f"  original_layer_12.pt: {orig_cls.shape}")
    print(f"  negated_layer_12.pt: {neg_cls.shape}")
    print(f"  original_token_ids.pt: {orig_ids.shape}")
    print(f"  negated_token_ids.pt: {neg_ids.shape}")

    return len(image_emb)


def main():
    parser = argparse.ArgumentParser(description="Extract negcap features for CLIP model")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="all")
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "metadata.json").exists():
        print(f"Already complete: {output_dir / 'metadata.json'} exists. Skipping.")
        return

    import open_clip
    print(f"Loading CLIP: {args.model} ({args.pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, cache_dir=CLIP_CACHE_DIR)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(args.model)

    from peakpatch.clip_utils import _get_text_encoder
    text_enc = _get_text_encoder(model)
    embed_dim = text_enc.ln_final.weight.shape[0]
    print(f"  embed_dim: {embed_dim}")

    print(f"Loading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    total = len(df)
    print(f"  Total samples: {total}")

    if args.split in ("train", "val"):
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        split_idx = int(args.split_ratio * total)
        if args.split == "train":
            df = df.iloc[:split_idx].reset_index(drop=True)
        else:
            df = df.iloc[split_idx:].reset_index(drop=True)
        print(f"  {args.split} split: {len(df)} samples")

    N = len(df)
    n_batches = (N + args.batch_size - 1) // args.batch_size

    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    existing_shards = sorted(shard_dir.glob("image_emb_*.pt"))
    start_shard = len(existing_shards)
    start_batch = start_shard * SHARD_SIZE
    if start_shard > 0:
        print(f"  Resuming: found {start_shard} shards, skipping first {start_batch} batches")

    buf_image, buf_orig_cls, buf_neg_cls, buf_orig_ids, buf_neg_ids = [], [], [], [], []
    shard_idx = start_shard
    batch_in_shard = 0
    n_skipped = 0

    def _load_image(row):
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            return preprocess(img)
        except Exception:
            return None

    def _load_batch(batch_df):
        """Load and filter a batch, returning (valid_imgs, valid_orig, valid_neg, n_bad)."""
        rows = [row for _, row in batch_df.iterrows()]
        results = list(pool.map(_load_image, rows))
        imgs, orig, neg, bad = [], [], [], 0
        for i, tensor in enumerate(results):
            if tensor is not None:
                imgs.append(tensor)
                row = batch_df.iloc[i]
                orig.append(str(row["original_caption"]))
                neg.append(str(row["negated_caption"]))
            else:
                bad += 1
        return imgs, orig, neg, bad

    batch_starts = list(range(0, N, args.batch_size))
    active_batches = [(i, s) for i, s in enumerate(batch_starts) if i >= start_batch]

    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    prefetch_pool = ThreadPoolExecutor(max_workers=1)

    if active_batches:
        first_num, first_start = active_batches[0]
        first_end = min(first_start + args.batch_size, N)
        prefetch_future = prefetch_pool.submit(_load_batch, df.iloc[first_start:first_end])

    for idx, (batch_num, start) in enumerate(tqdm(active_batches, desc="Extracting")):
        valid_imgs, valid_orig, valid_neg, n_bad = prefetch_future.result()
        n_skipped += n_bad

        if idx + 1 < len(active_batches):
            next_num, next_start = active_batches[idx + 1]
            next_end = min(next_start + args.batch_size, N)
            prefetch_future = prefetch_pool.submit(_load_batch, df.iloc[next_start:next_end])

        if not valid_imgs:
            batch_in_shard += 1
            if batch_in_shard >= SHARD_SIZE:
                if buf_image:
                    save_shard(shard_dir, shard_idx, buf_image, buf_orig_cls, buf_neg_cls, buf_orig_ids, buf_neg_ids)
                    print(f"  Saved shard {shard_idx} ({sum(x.shape[0] for x in buf_image)} samples)")
                    shard_idx += 1
                buf_image, buf_orig_cls, buf_neg_cls, buf_orig_ids, buf_neg_ids = [], [], [], [], []
                batch_in_shard = 0
            continue

        images = torch.stack(valid_imgs).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_emb = F.normalize(model.encode_image(images).float(), dim=-1)
        buf_image.append(image_emb.cpu())

        orig_cls, orig_ids = extract_text_l12_cls(valid_orig, model, tokenizer, device)
        buf_orig_cls.append(orig_cls.cpu())
        buf_orig_ids.append(orig_ids)

        neg_cls, neg_ids = extract_text_l12_cls(valid_neg, model, tokenizer, device)
        buf_neg_cls.append(neg_cls.cpu())
        buf_neg_ids.append(neg_ids)

        batch_in_shard += 1
        if batch_in_shard >= SHARD_SIZE:
            save_shard(shard_dir, shard_idx, buf_image, buf_orig_cls, buf_neg_cls, buf_orig_ids, buf_neg_ids)
            print(f"  Saved shard {shard_idx} ({sum(x.shape[0] for x in buf_image)} samples)")
            shard_idx += 1
            buf_image, buf_orig_cls, buf_neg_cls, buf_orig_ids, buf_neg_ids = [], [], [], [], []
            batch_in_shard = 0

    pool.shutdown()
    prefetch_pool.shutdown()

    if buf_image:
        save_shard(shard_dir, shard_idx, buf_image, buf_orig_cls, buf_neg_cls, buf_orig_ids, buf_neg_ids)
        print(f"  Saved shard {shard_idx} ({sum(x.shape[0] for x in buf_image)} samples)")
        shard_idx += 1

    print(f"\nSkipped {n_skipped} samples (missing images)")
    n_total = concat_shards(shard_dir, output_dir)

    metadata = {
        "num_samples": n_total,
        "selected_layers": [12],
        "embed_dim": embed_dim,
        "source_csv": str(args.input_csv),
        "dataset_type": "negcap",
        "clip_model": args.model,
        "clip_pretrained": args.pretrained,
        "split": args.split,
        "split_ratio": args.split_ratio,
        "seed": args.seed,
        "n_skipped": n_skipped,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Features saved to {output_dir}")


if __name__ == "__main__":
    main()
