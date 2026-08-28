"""Train EmbeddingCorrector with learned query (no negation detector needed).

Same pipeline as the original token DP training but without negator mask computation.
The model learns to discover negation-relevant positions via attention.

Usage:
    uv run python scripts/train_ec.py --exp-name tadp_v2_default
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peakpatch import EmbeddingCorrectorDataset
from peakpatch.clip_utils import compute_padding_mask, extract_token_sequences
from peakpatch.model import EmbeddingCorrector

CLIP_CACHE_DIR = os.environ.get("CLIP_CACHE_DIR") or None


def train_epoch(model, loader, optimizer, loss_type, grad_clip, device, clip_model,
                anchor_layer, target_layer):
    model.train()
    total_loss = 0
    total_cos = 0
    n = 0

    no_anchor = model.no_anchor
    all_layers = [target_layer] if no_anchor else [anchor_layer, target_layer]

    for token_ids, _neg_L12_cls, target in loader:
        token_ids = token_ids.to(device)
        target = target.to(device)

        with torch.no_grad():
            hidden = extract_token_sequences(token_ids, clip_model, layers=all_layers)
            padding_mask = compute_padding_mask(token_ids, clip_model)

        H_anchor = hidden[target_layer] if no_anchor else hidden[anchor_layer]
        H_target = hidden[target_layer]
        pred = model(H_anchor, H_target, padding_mask)

        if loss_type == "mse":
            loss = F.mse_loss(pred, target)
        elif loss_type == "cosine":
            loss = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        else:
            mse = F.mse_loss(pred, target)
            cos = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
            loss = mse + cos

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(token_ids)
        cos_sim = F.cosine_similarity(pred, target, dim=-1).mean().item()
        total_cos += cos_sim * len(token_ids)
        n += len(token_ids)

    return total_loss / n, total_cos / n


@torch.no_grad()
def validate(model, loader, loss_type, device, clip_model, anchor_layer, target_layer):
    model.eval()
    total_loss = 0
    total_cos = 0
    n = 0

    no_anchor = model.no_anchor
    all_layers = [target_layer] if no_anchor else [anchor_layer, target_layer]

    for token_ids, _neg_L12_cls, target in loader:
        token_ids = token_ids.to(device)
        target = target.to(device)

        hidden = extract_token_sequences(token_ids, clip_model, layers=all_layers)
        padding_mask = compute_padding_mask(token_ids, clip_model)

        H_anchor = hidden[target_layer] if no_anchor else hidden[anchor_layer]
        H_target = hidden[target_layer]
        pred = model(H_anchor, H_target, padding_mask)

        if loss_type == "mse":
            loss = F.mse_loss(pred, target)
        elif loss_type == "cosine":
            loss = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        else:
            mse = F.mse_loss(pred, target)
            cos = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
            loss = mse + cos

        total_loss += loss.item() * len(token_ids)
        cos_sim = F.cosine_similarity(pred, target, dim=-1).mean().item()
        total_cos += cos_sim * len(token_ids)
        n += len(token_ids)

    return total_loss / n, total_cos / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--loss-type", type=str, default="combined",
                        choices=["mse", "cosine", "combined"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--train-dir", type=str, default="data/negcap/train")
    parser.add_argument("--val-dir", type=str, default="data/negcap/val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="results/embedding_corrector")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--target-layer", type=int, default=12,
                        help="Target layer for cross-attention key/value (default: 12)")
    parser.add_argument("--anchor-layer", type=int, default=10,
                        help="Anchor layer for CLS input (default: 10)")
    parser.add_argument("--no-anchor", action="store_true",
                        help="Skip anchor CLS from input (3*D instead of 4*D)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Experiment: {args.exp_name}")
    print(f"Hidden dim: {args.hidden_dim}, Heads: {args.num_heads}")
    print(f"Loss: {args.loss_type}")
    print(f"Target layer: {args.target_layer}, Anchor layer: {args.anchor_layer}")
    print(f"No anchor: {args.no_anchor}")

    # Load frozen CLIP
    import open_clip
    from peakpatch.clip_utils import _get_text_encoder
    print(f"\nLoading frozen CLIP: {args.clip_model} ({args.clip_pretrained})")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained,
        cache_dir=CLIP_CACHE_DIR,
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Auto-detect embed_dim from CLIP model
    text_enc = _get_text_encoder(clip_model)
    embed_dim = text_enc.ln_final.weight.shape[0]
    print(f"  Detected embed_dim: {embed_dim}")

    # Load data
    print(f"\nLoading training data from {args.train_dir}...")
    train_ds = EmbeddingCorrectorDataset(args.train_dir, args.max_train_samples)
    print(f"  Train samples: {len(train_ds)}")

    print(f"Loading validation data from {args.val_dir}...")
    val_ds = EmbeddingCorrectorDataset(args.val_dir, max_samples=None)
    print(f"  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Create model
    model = EmbeddingCorrector(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=0.1,
        no_anchor=args.no_anchor,
    ).to(device)
    model_type = "embedding_corrector"

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model_type}: {num_params:,} parameters")

    config = {
        "model_type": model_type,
        "embed_dim": embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "dropout": 0.1,
        "no_anchor": args.no_anchor,
        "target_layer": args.target_layer,
        "anchor_layer": args.anchor_layer,
        "loss_type": args.loss_type,
        "num_parameters": num_params,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
    }
    out_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_cos = -1
    history = []
    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_cos = train_epoch(
            model, train_loader, optimizer, args.loss_type, args.grad_clip,
            device, clip_model, args.anchor_layer, args.target_layer)
        val_loss, val_cos = validate(
            model, val_loader, args.loss_type, device, clip_model,
            args.anchor_layer, args.target_layer)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        alpha = torch.exp(model.log_alpha).item()
        print(f"Epoch {epoch:2d}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_cos={train_cos:.4f} | val_cos={val_cos:.4f} | "
              f"alpha={alpha:.3f} | lr={lr:.6f}")

        epoch_record = {
            "epoch": epoch, "train_loss": train_loss, "train_cos": train_cos,
            "val_loss": val_loss, "val_cos": val_cos, "lr": lr, "alpha": alpha,
        }

        history.append(epoch_record)

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_dir / "best_model.pt")
            print(f"  >> Saved best model (val_cos={val_cos:.4f})")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, ckpt_dir / "final_model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val cosine: {best_val_cos:.4f}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
