"""Joint EC+SC training script."""

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peakpatch.clip_utils import _get_text_encoder, clip_text_forward_combined
from peakpatch.dataset import (
    MCQJointDataset,
    NegcapJointDataset,
    mcq_collate_fn,
    negcap_collate_fn,
)
from peakpatch.loss import JointLoss
from peakpatch.model import EmbeddingCorrector, ScoreCorrector

CLIP_CACHE_DIR = os.environ.get("CLIP_CACHE_DIR") or None

_NEGATION_TOKEN_IDS = None


def init_negation_tokens(tokenizer):
    """Detect negation token IDs from the CLIP/SigLIP tokenizer."""
    global _NEGATION_TOKEN_IDS
    neg_words = ["not", "no", "without", "never", "neither", "nor", "nothing", "none"]
    # Pad/BOS/EOS for both tokenizers; CLIP token 1 is '!', which never appears
    # when tokenizing negation words.
    special = {0, 1, 49406, 49407}
    token_set = set()
    for word in neg_words:
        ids = tokenizer([word])[0]
        for t in ids.tolist():
            if t not in special:
                token_set.add(t)
    _NEGATION_TOKEN_IDS = torch.tensor(sorted(token_set))


def detect_negation(token_ids: torch.Tensor) -> torch.Tensor:
    """Detect which token sequences contain negation words."""
    B, N_opts, S = token_ids.shape
    flat = token_ids.view(-1, S)
    neg_ids = _NEGATION_TOKEN_IDS.to(token_ids.device)
    match = (flat.unsqueeze(-1) == neg_ids.unsqueeze(0).unsqueeze(0))
    has_neg = match.any(dim=-1).any(dim=-1)
    return has_neg.view(B, N_opts)


def ec_step(
    ec_model: EmbeddingCorrector,
    batch: Dict[str, torch.Tensor],
    clip_model,
    target_layer: int,
    anchor_layer: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for EC on negcap batch. Returns S_orig [B,B] and S_neg [B,B]."""
    image_emb = batch["image_emb"].to(device)
    orig_cls = batch["orig_L12_cls"].to(device)
    neg_cls = batch["neg_L12_cls"].to(device)
    orig_ids = batch["orig_token_ids"].to(device)
    neg_ids = batch["neg_token_ids"].to(device)

    with torch.no_grad():
        orig_out = clip_text_forward_combined(
            orig_ids, clip_model, target_layer, anchor_layer)
    orig_corrected, _ = ec_model.correct_embedding(
        orig_cls, orig_out["H_anchor"], orig_out["H_target"],
        orig_out["padding_mask"])

    with torch.no_grad():
        neg_out = clip_text_forward_combined(
            neg_ids, clip_model, target_layer, anchor_layer)
    neg_corrected, _ = ec_model.correct_embedding(
        neg_cls, neg_out["H_anchor"], neg_out["H_target"],
        neg_out["padding_mask"])

    S_orig = image_emb @ orig_corrected.T
    S_neg = image_emb @ neg_corrected.T
    return S_orig, S_neg


def sc_step(
    ec_model: EmbeddingCorrector,
    sc_model: ScoreCorrector,
    batch: Dict[str, torch.Tensor],
    clip_model,
    target_layer: int,
    anchor_layer: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for SC on MCQ batch. Returns mcq_scores, corrections, labels."""
    image_emb = batch["image_emb"].to(device)
    mcq_layers = {
        layer: feats.to(device)
        for layer, feats in batch["mcq_layers"].items()
    }
    mcq_token_ids = batch["mcq_token_ids"].to(device)
    mcq_labels = batch["mcq_label"].to(device)

    B = image_emb.shape[0]
    sc_layers = sc_model.selected_layers
    anchor_sc_layer = max(sc_layers)
    need_tokens = sc_model.use_token_features

    corrected_L12_list = []
    token_seqs_list = []
    padding_masks_list = []
    for opt_idx in range(4):
        opt_ids = mcq_token_ids[:, opt_idx, :]
        opt_L12 = mcq_layers[anchor_sc_layer][:, opt_idx, :]

        with torch.no_grad():
            clip_out = clip_text_forward_combined(
                opt_ids, clip_model, target_layer, anchor_layer)

        corrected, _ = ec_model.correct_embedding(
            opt_L12, clip_out["H_anchor"], clip_out["H_target"],
            clip_out["padding_mask"])
        corrected_L12_list.append(corrected)

        if need_tokens:
            token_seqs_list.append(clip_out["H_target"])
            padding_masks_list.append(clip_out["padding_mask"])

    corrected_L12 = torch.stack(corrected_L12_list, dim=1)

    sc_mcq_layers = {}
    for layer in sc_layers:
        if layer == anchor_sc_layer:
            sc_mcq_layers[layer] = corrected_L12
        else:
            sc_mcq_layers[layer] = mcq_layers[layer]

    mcq_token_seqs = None
    mcq_padding_masks = None
    if need_tokens:
        mcq_token_seqs = torch.stack(token_seqs_list, dim=1)
        mcq_padding_masks = torch.stack(padding_masks_list, dim=1)

    mcq_scores, mcq_corrections = sc_model.score_mcq(
        sc_mcq_layers, image_emb, mcq_token_seqs, mcq_padding_masks)
    return mcq_scores, mcq_corrections, mcq_labels, mcq_token_ids


def train_epoch(
    ec_model: EmbeddingCorrector,
    sc_model: ScoreCorrector,
    negcap_loader: DataLoader,
    mcq_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: JointLoss,
    clip_model,
    target_layer: int,
    anchor_layer: int,
    grad_clip: float,
    device: str,
) -> Dict[str, float]:
    """Train one epoch. MCQ loader defines epoch length, negcap cycles."""
    ec_model.train()
    sc_model.train()

    negcap_iter = itertools.cycle(negcap_loader)

    totals = {}
    n_steps = 0

    for mcq_batch in mcq_loader:
        negcap_batch = next(negcap_iter)

        S_orig, S_neg = ec_step(
            ec_model, negcap_batch, clip_model, target_layer, anchor_layer, device)

        mcq_scores, mcq_corrections, mcq_labels, mcq_token_ids = sc_step(
            ec_model, sc_model, mcq_batch, clip_model,
            target_layer, anchor_layer, device)

        neg_mask = detect_negation(mcq_token_ids) if _NEGATION_TOKEN_IDS is not None else None

        loss, metrics = loss_fn(
            S_orig, S_neg, mcq_scores, mcq_corrections, mcq_labels, neg_mask)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            all_params = list(ec_model.parameters()) + list(sc_model.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n_steps += 1

    return {k: v / n_steps for k, v in totals.items()}


@torch.no_grad()
def validate(
    ec_model: EmbeddingCorrector,
    sc_model: ScoreCorrector,
    negcap_loader: DataLoader,
    mcq_loader: DataLoader,
    loss_fn: JointLoss,
    clip_model,
    target_layer: int,
    anchor_layer: int,
    device: str,
) -> Dict[str, float]:
    """Validate on both datasets."""
    ec_model.eval()
    sc_model.eval()

    negcap_iter = itertools.cycle(negcap_loader)

    totals = {}
    n_steps = 0

    for mcq_batch in mcq_loader:
        negcap_batch = next(negcap_iter)

        S_orig, S_neg = ec_step(
            ec_model, negcap_batch, clip_model, target_layer, anchor_layer, device)

        mcq_scores, mcq_corrections, mcq_labels, mcq_token_ids = sc_step(
            ec_model, sc_model, mcq_batch, clip_model,
            target_layer, anchor_layer, device)

        neg_mask = detect_negation(mcq_token_ids) if _NEGATION_TOKEN_IDS is not None else None
        _, metrics = loss_fn(
            S_orig, S_neg, mcq_scores, mcq_corrections, mcq_labels, neg_mask)

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n_steps += 1

    return {k: v / n_steps for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Joint EC+SC training")
    parser.add_argument("--negcap-train-dir", type=str, required=True)
    parser.add_argument("--negcap-val-dir", type=str, required=True)
    parser.add_argument("--mcq-train-dir", type=str, required=True)
    parser.add_argument("--mcq-val-dir", type=str, required=True)
    parser.add_argument("--max-negcap-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ec-batch-size", type=int, default=256)
    parser.add_argument("--sc-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="SC learning rate (EC uses lr * ec-lr-scale)")
    parser.add_argument("--ec-lr-scale", type=float, default=0.1,
                        help="EC lr multiplier relative to SC lr")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--sc-temperature", type=float, default=0.07)
    parser.add_argument("--lambda-sc", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--lambda-reg-nonneg", type=float, default=0.0,
                        help="Extra regularization for non-negation corrections")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--anchor-layer", type=int, default=6)
    parser.add_argument("--sc-layers", type=int, nargs="+", default=[6, 8, 12])
    parser.add_argument("--sc-max-correction", type=float, default=0.20,
                        help="Bound on the SC score offset")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/joint")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--ec-no-anchor", action="store_true",
                        help="EC: skip anchor layer, use only target layer features")
    parser.add_argument("--ec-cls-only", action="store_true",
                        help="EC ablation: use only CLS tokens, no cross-attention/pooling")
    parser.add_argument("--ec-no-pool", action="store_true",
                        help="EC ablation: drop mean-pooled embedding from MLP input")
    parser.add_argument("--sc-use-tokens", action="store_true",
                        help="SC ablation: add token-level cross-attention features")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a joint checkpoint (best_joint.pt or final_joint.pt)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Experiment: {args.exp_name}")

    import open_clip
    print(f"\nLoading frozen CLIP: {args.clip_model} ({args.clip_pretrained})")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained,
        cache_dir=CLIP_CACHE_DIR,
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    text_enc = _get_text_encoder(clip_model)
    clip_embed_dim = text_enc.ln_final.weight.shape[0]
    print(f"  CLIP embed_dim: {clip_embed_dim}")

    tokenizer = open_clip.get_tokenizer(args.clip_model)
    init_negation_tokens(tokenizer)
    print(f"  Negation token IDs: {_NEGATION_TOKEN_IDS.tolist()}")

    ec_model = EmbeddingCorrector(
        embed_dim=clip_embed_dim,
        no_anchor=args.ec_no_anchor,
        cls_only=args.ec_cls_only,
        no_pool=args.ec_no_pool,
    ).to(device)
    ec_model.train()
    for p in ec_model.parameters():
        p.requires_grad = True
    ec_alpha = torch.exp(ec_model.log_alpha).item()
    print(f"\nEC params: {ec_model.count_parameters():,}, alpha={ec_alpha:.4f}"
          f", cls_only={ec_model.cls_only}")

    sc_model = ScoreCorrector(
        embed_dim=clip_embed_dim,
        max_correction=args.sc_max_correction,
        selected_layers=args.sc_layers,
        use_token_features=args.sc_use_tokens,
    ).to(device)
    sc_model.train()
    for p in sc_model.parameters():
        p.requires_grad = True
    print(f"SC params: {sc_model.count_parameters():,}, layers={sc_model.selected_layers}"
          f", use_token_features={sc_model.use_token_features}")

    print(f"\nLoading negcap train: {args.negcap_train_dir}")
    negcap_train = NegcapJointDataset(args.negcap_train_dir, args.max_negcap_samples)
    print(f"Loading negcap val: {args.negcap_val_dir}")
    negcap_val = NegcapJointDataset(args.negcap_val_dir)

    print(f"Loading MCQ train: {args.mcq_train_dir}")
    mcq_train = MCQJointDataset(args.mcq_train_dir, layers=args.sc_layers)
    print(f"Loading MCQ val: {args.mcq_val_dir}")
    mcq_val = MCQJointDataset(args.mcq_val_dir, layers=args.sc_layers, shuffle_mcq=False)

    negcap_train_loader = DataLoader(
        negcap_train, batch_size=args.ec_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=negcap_collate_fn)
    negcap_val_loader = DataLoader(
        negcap_val, batch_size=args.ec_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=negcap_collate_fn)
    mcq_train_loader = DataLoader(
        mcq_train, batch_size=args.sc_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=mcq_collate_fn)
    mcq_val_loader = DataLoader(
        mcq_val, batch_size=args.sc_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=mcq_collate_fn)

    ec_lr = args.lr * args.ec_lr_scale
    optimizer = torch.optim.AdamW([
        {"params": ec_model.parameters(), "lr": ec_lr},
        {"params": sc_model.parameters(), "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    loss_fn = JointLoss(
        logit_scale=args.logit_scale,
        sc_temperature=args.sc_temperature,
        lambda_sc=args.lambda_sc,
        lambda_reg=args.lambda_reg,
        lambda_reg_nonneg=args.lambda_reg_nonneg,
    )

    out_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "target_layer": args.target_layer,
        "anchor_layer": args.anchor_layer,
        "sc_layers": args.sc_layers,
        "sc_max_correction": args.sc_max_correction,
        "epochs": args.epochs,
        "ec_batch_size": args.ec_batch_size,
        "sc_batch_size": args.sc_batch_size,
        "lr": args.lr,
        "ec_lr_scale": args.ec_lr_scale,
        "weight_decay": args.weight_decay,
        "logit_scale": args.logit_scale,
        "sc_temperature": args.sc_temperature,
        "lambda_sc": args.lambda_sc,
        "lambda_reg": args.lambda_reg,
        "lambda_reg_nonneg": args.lambda_reg_nonneg,
        "negcap_train_samples": len(negcap_train),
        "mcq_train_samples": len(mcq_train),
        "negcap_val_samples": len(negcap_val),
        "mcq_val_samples": len(mcq_val),
        "ec_params": ec_model.count_parameters(),
        "sc_params": sc_model.count_parameters(),
        "ec_cls_only": args.ec_cls_only,
        "ec_no_anchor": args.ec_no_anchor,
        "ec_no_pool": args.ec_no_pool,
        "sc_use_tokens": args.sc_use_tokens,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    mcq_steps_per_epoch = len(mcq_train_loader)
    print(f"\nTraining config:")
    print(f"  EC lr: {ec_lr:.2e}, SC lr: {args.lr:.2e}")
    print(f"  MCQ steps/epoch: {mcq_steps_per_epoch}")
    print(f"  Negcap samples/epoch: {mcq_steps_per_epoch * args.ec_batch_size}")
    print(f"  lambda_sc: {args.lambda_sc}, logit_scale: {args.logit_scale}"
          f", lambda_reg_nonneg: {args.lambda_reg_nonneg}")

    best_val_sc_acc = 0.0
    history = []
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoints" / "best_joint.pt"
        if resume_path.exists():
            print(f"\nResuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            ec_model.load_state_dict(ckpt["ec_state_dict"])
            sc_model.load_state_dict(ckpt["sc_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if "val_metrics" in ckpt:
                best_val_sc_acc = ckpt["val_metrics"].get("sc_acc", 0.0)
            for _ in range(start_epoch - 1):
                scheduler.step()
            history_path = out_dir / "history.json"
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
            print(f"  Resumed at epoch {start_epoch}, best_val_sc_acc={best_val_sc_acc:.4f}")
        else:
            print(f"  Resume checkpoint not found: {resume_path}, starting fresh")

    if start_epoch > args.epochs:
        print(f"Already completed {args.epochs} epochs. Nothing to do.")
        return

    print(f"\nTraining epochs {start_epoch}-{args.epochs}...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            ec_model, sc_model, negcap_train_loader, mcq_train_loader,
            optimizer, loss_fn, clip_model,
            args.target_layer, args.anchor_layer, args.grad_clip, device)

        val_metrics = validate(
            ec_model, sc_model, negcap_val_loader, mcq_val_loader,
            loss_fn, clip_model, args.target_layer, args.anchor_layer, device)

        scheduler.step()
        elapsed = time.time() - t0

        ec_lr_now = optimizer.param_groups[0]["lr"]
        sc_lr_now = optimizer.param_groups[1]["lr"]
        ec_alpha = torch.exp(ec_model.log_alpha).item()

        print(
            f"Epoch {epoch:2d}/{args.epochs} ({elapsed:.0f}s) | "
            f"loss={train_metrics['loss_total']:.4f} | "
            f"ec_i2t={train_metrics['ec_acc_i2t']:.3f} "
            f"sc_acc={train_metrics['sc_acc']:.3f} | "
            f"val_loss={val_metrics['loss_total']:.4f} "
            f"val_sc={val_metrics['sc_acc']:.3f} "
            f"val_ec_i2t={val_metrics['ec_acc_i2t']:.3f} | "
            f"corr neg={train_metrics.get('sc_corr_neg', 0):.4f} "
            f"nonneg={train_metrics.get('sc_corr_nonneg', 0):.4f} | "
            f"alpha={ec_alpha:.3f}"
        )

        record = {
            "epoch": epoch, "elapsed": elapsed,
            "ec_alpha": ec_alpha, "ec_lr": ec_lr_now, "sc_lr": sc_lr_now,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if val_metrics["sc_acc"] > best_val_sc_acc:
            best_val_sc_acc = val_metrics["sc_acc"]
            torch.save({
                "epoch": epoch,
                "ec_state_dict": ec_model.state_dict(),
                "sc_state_dict": sc_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
            }, ckpt_dir / "best_joint.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": ec_model.state_dict(),
                "config": {
                    "model_type": "embedding_corrector",
                    "clip_model": args.clip_model,
                    "clip_pretrained": args.clip_pretrained,
                    "embed_dim": ec_model.embed_dim,
                    "hidden_dim": ec_model.hidden_dim,
                    "num_heads": ec_model.num_heads,
                    "dropout": ec_model.dropout_rate,
                    "no_anchor": ec_model.no_anchor,
                    "cls_only": ec_model.cls_only,
                    "no_pool": ec_model.no_pool,
                    "target_layer": args.target_layer,
                    "anchor_layer": args.anchor_layer,
                },
            }, ckpt_dir / "best_ec.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": sc_model.state_dict(),
                "config": {
                    "clip_model": args.clip_model,
                    "clip_pretrained": args.clip_pretrained,
                    "embed_dim": sc_model.embed_dim,
                    "context_dim": sc_model.context_dim,
                    "max_correction": sc_model.max_correction,
                    "layers": sc_model.selected_layers,
                    "use_gate": sc_model.use_gate,
                    "correction_mode": sc_model.correction_mode,
                    "use_token_features": sc_model.use_token_features,
                    "token_num_heads": sc_model.token_num_heads,
                },
            }, ckpt_dir / "best_sc.pt")
            print(f"  >> Saved best (val_sc_acc={val_metrics['sc_acc']:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": ec_model.state_dict(),
            "config": {
                "model_type": "embedding_corrector",
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "embed_dim": ec_model.embed_dim,
                "hidden_dim": ec_model.hidden_dim,
                "num_heads": ec_model.num_heads,
                "dropout": ec_model.dropout_rate,
                "no_anchor": ec_model.no_anchor,
                "cls_only": ec_model.cls_only,
                "target_layer": args.target_layer,
                "anchor_layer": args.anchor_layer,
            },
        }, ckpt_dir / "latest_ec.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": sc_model.state_dict(),
            "config": {
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "embed_dim": sc_model.embed_dim,
                "context_dim": sc_model.context_dim,
                "max_correction": sc_model.max_correction,
                "layers": sc_model.selected_layers,
                "use_gate": sc_model.use_gate,
                "correction_mode": sc_model.correction_mode,
                "use_token_features": sc_model.use_token_features,
                "token_num_heads": sc_model.token_num_heads,
            },
        }, ckpt_dir / "latest_sc.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save({
        "epoch": args.epochs,
        "ec_state_dict": ec_model.state_dict(),
        "sc_state_dict": sc_model.state_dict(),
        "config": config,
    }, ckpt_dir / "final_joint.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val SC accuracy: {best_val_sc_acc:.4f}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
