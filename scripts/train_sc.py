"""Training script for ScoreCorrector model."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from peakpatch import ScoreCorrector, ScoreCorrectorLoss, NegBenchDataset, collate_fn


def sanity_check_contrastive(model: ScoreCorrector, dataloader: DataLoader, device: str):
    """Verify score_contrastive matches score_pairwise on a small batch."""
    model.eval()
    batch = next(iter(dataloader))
    text_layers = {k: v.to(device) for k, v in batch["text_layers"].items()}
    image_emb = batch["image_emb"].to(device)

    with torch.no_grad():
        scores_c, _ = model.score_contrastive(text_layers, image_emb)

        scores_p = model.score_pairwise(text_layers, image_emb, batch_size=256)

    max_diff = (scores_c - scores_p).abs().max().item()
    print(f"[Sanity] score_contrastive vs score_pairwise max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, (
        f"score_contrastive and score_pairwise disagree: max_diff={max_diff:.2e}"
    )
    print("[Sanity] PASSED")
    model.train()


def _extract_neg_layers(
    mcq_layers: Dict[int, torch.Tensor],
    mcq_labels: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """Extract a random wrong MCQ option per sample as negated text features."""
    B = mcq_labels.shape[0]
    device = mcq_labels.device

    wrong_mask = torch.ones(B, 4, dtype=torch.bool, device=device)
    wrong_mask[torch.arange(B, device=device), mcq_labels] = False

    rand_within_wrong = torch.randint(3, (B,), device=device)
    wrong_indices = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(B):
        wrong_opts = torch.where(wrong_mask[i])[0]
        wrong_indices[i] = wrong_opts[rand_within_wrong[i]]

    return {
        layer: mcq_layers[layer][torch.arange(B, device=device), wrong_indices]
        for layer in mcq_layers
    }


def train_epoch(
    model: ScoreCorrector,
    loss_fn: ScoreCorrectorLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    loss_type: str = "mcq",
    grad_clip: float = 1.0,
    log_every: int = 100,
    epoch: int = 0,
    global_step: int = 0,
    use_wandb: bool = False,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    ema_params: Optional[Dict[str, torch.Tensor]] = None,
    ema_decay: float = 0.0,
    alpha_anti: float = 0.0,
    anti_mode: str = "hinge",
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mcq_loss = 0.0
    total_contrastive_loss = 0.0
    total_anti_loss = 0.0
    total_reg_loss = 0.0
    total_mcq_acc = 0.0
    total_baseline_acc = 0.0
    total_mean_corr = 0.0
    total_max_corr = 0.0
    total_acc_i2t = 0.0
    total_acc_t2i = 0.0
    total_anti_acc = 0.0
    total_anti_violations = 0.0
    total_anti_diag_rank = 0.0
    total_neg_above_pos = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch_idx, batch in enumerate(pbar):
        text_layers = {k: v.to(device) for k, v in batch["text_layers"].items()}
        image_emb = batch["image_emb"].to(device)
        mcq_layers = {k: v.to(device) for k, v in batch["mcq_layers"].items()}
        mcq_labels = batch["mcq_label"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if loss_type == "mcq":
                mcq_scores, mcq_corrections = model.score_mcq(mcq_layers, image_emb)
                loss, loss_dict = loss_fn(mcq_scores, mcq_corrections, mcq_labels)
                mcq_acc = loss_fn.compute_mcq_accuracy(mcq_scores, mcq_labels)
                baseline_acc = loss_fn.compute_baseline_accuracy(
                    mcq_layers, image_emb, mcq_labels,
                    anchor_layer=max(model.selected_layers),
                )
                loss_dict["mcq_accuracy"] = mcq_acc
                loss_dict["baseline_accuracy"] = baseline_acc

            elif loss_type == "contrastive":
                scores, corrections = model.score_contrastive(text_layers, image_emb)
                loss, loss_dict = loss_fn.compute_contrastive_loss(scores, corrections)

            elif loss_type == "combined":
                scores, corrections = model.score_contrastive(text_layers, image_emb)
                mcq_scores, mcq_corrections = model.score_mcq(mcq_layers, image_emb)

                anti_scores = None
                anti_corrections = None
                if alpha_anti > 0 or anti_mode == "expanded":
                    neg_layers = _extract_neg_layers(mcq_layers, mcq_labels)
                    anti_scores, anti_corrections = model.score_contrastive(
                        neg_layers, image_emb,
                    )

                loss, loss_dict = loss_fn.forward_combined(
                    scores, corrections, mcq_scores, mcq_corrections, mcq_labels,
                    anti_scores=anti_scores,
                    anti_corrections=anti_corrections,
                )
                baseline_acc = loss_fn.compute_baseline_accuracy(
                    mcq_layers, image_emb, mcq_labels,
                    anchor_layer=max(model.selected_layers),
                )
                loss_dict["baseline_accuracy"] = baseline_acc

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        is_accumulation_step = (batch_idx + 1) % grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)

        if is_accumulation_step or is_last_batch:
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if ema_params is not None and ema_decay > 0:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in ema_params:
                            ema_params[name].mul_(ema_decay).add_(
                                param.data, alpha=1.0 - ema_decay
                            )

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            global_step += 1

        total_loss += loss_dict["loss_total"]
        total_reg_loss += loss_dict["loss_reg"]
        total_mean_corr += loss_dict["mean_abs_correction"]
        total_max_corr += loss_dict["max_abs_correction"]

        if "loss_mcq" in loss_dict:
            total_mcq_loss += loss_dict["loss_mcq"]
        if "loss_contrastive" in loss_dict:
            total_contrastive_loss += loss_dict["loss_contrastive"]
        if "mcq_accuracy" in loss_dict:
            total_mcq_acc += loss_dict["mcq_accuracy"]
        if "baseline_accuracy" in loss_dict:
            total_baseline_acc += loss_dict["baseline_accuracy"]
        if "contrastive_acc_i2t" in loss_dict:
            total_acc_i2t += loss_dict["contrastive_acc_i2t"]
            total_acc_t2i += loss_dict["contrastive_acc_t2i"]
        if "loss_anti" in loss_dict:
            total_anti_loss += loss_dict["loss_anti"]
        if "anti_accuracy" in loss_dict:
            total_anti_acc += loss_dict["anti_accuracy"]
        if "anti_violations" in loss_dict:
            total_anti_violations += loss_dict["anti_violations"]
        if "anti_mean_diag_rank" in loss_dict:
            total_anti_diag_rank += loss_dict["anti_mean_diag_rank"]
        if "neg_above_pos" in loss_dict:
            total_neg_above_pos += loss_dict["neg_above_pos"]

        num_batches += 1

        if (is_accumulation_step or is_last_batch) and use_wandb and WANDB_AVAILABLE and global_step % log_every == 0:
            step_log = {
                "step": global_step,
                "step/loss_total": loss_dict["loss_total"],
                "step/loss_reg": loss_dict["loss_reg"],
                "step/mean_abs_correction": loss_dict["mean_abs_correction"],
                "step/max_abs_correction": loss_dict["max_abs_correction"],
                "step/learning_rate": current_lr,
                "step/grad_norm": grad_norm,
            }
            if "loss_mcq" in loss_dict:
                step_log["step/loss_mcq"] = loss_dict["loss_mcq"]
            if "loss_contrastive" in loss_dict:
                step_log["step/loss_contrastive"] = loss_dict["loss_contrastive"]
            if "loss_anti" in loss_dict:
                step_log["step/loss_anti"] = loss_dict["loss_anti"]
            if "mcq_accuracy" in loss_dict:
                step_log["step/mcq_accuracy"] = loss_dict["mcq_accuracy"]
            if "baseline_accuracy" in loss_dict:
                step_log["step/baseline_accuracy"] = loss_dict["baseline_accuracy"]
            if "contrastive_acc_i2t" in loss_dict:
                step_log["step/contrastive_acc_i2t"] = loss_dict["contrastive_acc_i2t"]
                step_log["step/contrastive_acc_t2i"] = loss_dict["contrastive_acc_t2i"]
            if "anti_accuracy" in loss_dict:
                step_log["step/anti_accuracy"] = loss_dict["anti_accuracy"]
            if "anti_violations" in loss_dict:
                step_log["step/anti_violations"] = loss_dict["anti_violations"]
            if "anti_mean_diag_rank" in loss_dict:
                step_log["step/anti_mean_diag_rank"] = loss_dict["anti_mean_diag_rank"]
            if "neg_above_pos" in loss_dict:
                step_log["step/neg_above_pos"] = loss_dict["neg_above_pos"]
            wandb.log(step_log, step=global_step)

        if loss_type == "mcq":
            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "mcq": f"{loss_dict['mcq_accuracy']:.3f}",
                "base": f"{loss_dict['baseline_accuracy']:.3f}",
                "|c|": f"{loss_dict['mean_abs_correction']:.4f}",
            })
        elif loss_type == "contrastive":
            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "i2t": f"{loss_dict['contrastive_acc_i2t']:.3f}",
                "t2i": f"{loss_dict['contrastive_acc_t2i']:.3f}",
                "|c|": f"{loss_dict['mean_abs_correction']:.4f}",
            })
        elif loss_type == "combined":
            postfix = {
                "loss": f"{loss_dict['loss_total']:.4f}",
                "mcq": f"{loss_dict['mcq_accuracy']:.3f}",
                "i2t": f"{loss_dict['contrastive_acc_i2t']:.3f}",
                "|c|": f"{loss_dict['mean_abs_correction']:.4f}",
            }
            if "anti_accuracy" in loss_dict:
                postfix["anti"] = f"{loss_dict['anti_accuracy']:.3f}"
            if "neg_above_pos" in loss_dict:
                postfix["neg>pos"] = f"{loss_dict['neg_above_pos']:.3f}"
            pbar.set_postfix(postfix)

    metrics = {
        "loss_total": total_loss / num_batches,
        "loss_reg": total_reg_loss / num_batches,
        "mean_abs_correction": total_mean_corr / num_batches,
        "max_abs_correction": total_max_corr / num_batches,
        "global_step": global_step,
    }
    if total_mcq_loss > 0:
        metrics["loss_mcq"] = total_mcq_loss / num_batches
    if total_contrastive_loss > 0:
        metrics["loss_contrastive"] = total_contrastive_loss / num_batches
    if total_anti_loss > 0 or alpha_anti > 0:
        metrics["loss_anti"] = total_anti_loss / max(num_batches, 1)
        metrics["anti_accuracy"] = total_anti_acc / max(num_batches, 1)
        metrics["anti_violations"] = total_anti_violations / max(num_batches, 1)
        metrics["anti_mean_diag_rank"] = total_anti_diag_rank / max(num_batches, 1)
    if total_neg_above_pos > 0 or anti_mode == "expanded":
        metrics["neg_above_pos"] = total_neg_above_pos / max(num_batches, 1)
    if loss_type in ("mcq", "combined"):
        metrics["mcq_accuracy"] = total_mcq_acc / num_batches
        metrics["baseline_accuracy"] = total_baseline_acc / num_batches
    if loss_type in ("contrastive", "combined"):
        metrics["contrastive_acc_i2t"] = total_acc_i2t / num_batches
        metrics["contrastive_acc_t2i"] = total_acc_t2i / num_batches

    return metrics


@torch.no_grad()
def evaluate(
    model: ScoreCorrector,
    loss_fn: ScoreCorrectorLoss,
    dataloader: DataLoader,
    device: str,
    loss_type: str = "mcq",
    anti_mode: str = "hinge",
) -> Dict[str, float]:
    """Evaluate the model on a validation set."""
    model.eval()

    total_loss = 0.0
    total_mcq_loss = 0.0
    total_contrastive_loss = 0.0
    total_reg_loss = 0.0
    total_mcq_acc = 0.0
    total_baseline_acc = 0.0
    total_mean_corr = 0.0
    total_acc_i2t = 0.0
    total_acc_t2i = 0.0
    total_neg_above_pos = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        text_layers = {k: v.to(device) for k, v in batch["text_layers"].items()}
        image_emb = batch["image_emb"].to(device)
        mcq_layers = {k: v.to(device) for k, v in batch["mcq_layers"].items()}
        mcq_labels = batch["mcq_label"].to(device)

        if loss_type == "mcq":
            mcq_scores, mcq_corrections = model.score_mcq(mcq_layers, image_emb)
            loss, loss_dict = loss_fn(mcq_scores, mcq_corrections, mcq_labels)
            mcq_acc = loss_fn.compute_mcq_accuracy(mcq_scores, mcq_labels)
            baseline_acc = loss_fn.compute_baseline_accuracy(
                mcq_layers, image_emb, mcq_labels,
                anchor_layer=max(model.selected_layers),
            )
            loss_dict["mcq_accuracy"] = mcq_acc
            loss_dict["baseline_accuracy"] = baseline_acc

        elif loss_type == "contrastive":
            scores, corrections = model.score_contrastive(text_layers, image_emb)
            loss, loss_dict = loss_fn.compute_contrastive_loss(scores, corrections)

        elif loss_type == "combined":
            scores, corrections = model.score_contrastive(text_layers, image_emb)
            mcq_scores, mcq_corrections = model.score_mcq(mcq_layers, image_emb)

            anti_scores = None
            anti_corrections = None
            if anti_mode == "expanded":
                neg_layers = _extract_neg_layers(mcq_layers, mcq_labels)
                anti_scores, anti_corrections = model.score_contrastive(
                    neg_layers, image_emb,
                )

            loss, loss_dict = loss_fn.forward_combined(
                scores, corrections, mcq_scores, mcq_corrections, mcq_labels,
                anti_scores=anti_scores,
                anti_corrections=anti_corrections,
            )
            baseline_acc = loss_fn.compute_baseline_accuracy(
                mcq_layers, image_emb, mcq_labels,
                anchor_layer=max(model.selected_layers),
            )
            loss_dict["baseline_accuracy"] = baseline_acc

        total_loss += loss_dict["loss_total"]
        total_reg_loss += loss_dict["loss_reg"]
        total_mean_corr += loss_dict["mean_abs_correction"]

        if "loss_mcq" in loss_dict:
            total_mcq_loss += loss_dict["loss_mcq"]
        if "loss_contrastive" in loss_dict:
            total_contrastive_loss += loss_dict["loss_contrastive"]
        if "mcq_accuracy" in loss_dict:
            total_mcq_acc += loss_dict["mcq_accuracy"]
        if "baseline_accuracy" in loss_dict:
            total_baseline_acc += loss_dict["baseline_accuracy"]
        if "contrastive_acc_i2t" in loss_dict:
            total_acc_i2t += loss_dict["contrastive_acc_i2t"]
            total_acc_t2i += loss_dict["contrastive_acc_t2i"]
        if "neg_above_pos" in loss_dict:
            total_neg_above_pos += loss_dict["neg_above_pos"]

        num_batches += 1

    metrics = {
        "loss_total": total_loss / num_batches,
        "loss_reg": total_reg_loss / num_batches,
        "mean_abs_correction": total_mean_corr / num_batches,
    }
    if total_mcq_loss > 0:
        metrics["loss_mcq"] = total_mcq_loss / num_batches
    if total_contrastive_loss > 0:
        metrics["loss_contrastive"] = total_contrastive_loss / num_batches
    if loss_type in ("mcq", "combined"):
        metrics["mcq_accuracy"] = total_mcq_acc / num_batches
        metrics["baseline_accuracy"] = total_baseline_acc / num_batches
    if loss_type in ("contrastive", "combined"):
        metrics["contrastive_acc_i2t"] = total_acc_i2t / num_batches
        metrics["contrastive_acc_t2i"] = total_acc_t2i / num_batches
    if total_neg_above_pos > 0 or anti_mode == "expanded":
        metrics["neg_above_pos"] = total_neg_above_pos / max(num_batches, 1)

    return metrics


def save_checkpoint(
    model: ScoreCorrector,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    path: str,
    config: Dict,
    ema_params: Optional[Dict[str, torch.Tensor]] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }
    if ema_params is not None:
        checkpoint["ema_params"] = {k: v.cpu() for k, v in ema_params.items()}
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: ScoreCorrector, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    loaded_ema = checkpoint.get("ema_params", None)
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {}), loaded_ema


def main():
    parser = argparse.ArgumentParser(
        description="Train ScoreCorrector model for score-level negation correction"
    )

    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory with pre-extracted training features",
    )
    parser.add_argument(
        "--val-dir", type=str, default=None,
        help="Directory with pre-extracted validation features",
    )

    parser.add_argument(
        "--layers", type=int, nargs="+", default=[3, 8, 12],
        help="CLIP layers to use",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=512,
        help="CLIP embedding dimension",
    )
    parser.add_argument(
        "--context-dim", type=int, default=128,
        help="Context vector dimension",
    )
    parser.add_argument(
        "--max-correction", type=float, default=0.15,
        help="Maximum absolute correction magnitude",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability",
    )
    parser.add_argument(
        "--use-gate", action="store_true",
        help="Add negation gate to suppress corrections on non-negation text",
    )
    parser.add_argument(
        "--correction-mode", type=str, default="fixed",
        choices=["fixed", "margin_relative", "cross_gate", "multiplicative", "two_head", "embedding"],
        help="Correction mode: how corrections are applied to baseline similarity",
    )
    parser.add_argument(
        "--margin-alpha", type=float, default=0.5,
        help="Fraction of cross-layer spread allowed as correction (margin_relative mode)",
    )
    parser.add_argument(
        "--mult-alpha", type=float, default=0.3,
        help="Max multiplicative deviation from 1.0 (multiplicative mode)",
    )
    parser.add_argument(
        "--confidence-gate", action="store_true",
        help="Scale corrections by inverse CLIP confidence (AMU-Tuning style). "
        "Suppresses corrections when CLIP is confident (classification), "
        "amplifies when uncertain (negation). Orthogonal to correction mode.",
    )
    parser.add_argument(
        "--confidence-rho", type=float, default=1.0,
        help="Exponent for confidence gating sharpness (higher = stronger suppression)",
    )
    parser.add_argument(
        "--embed-alpha", type=float, default=0.1,
        help="Maximum L2 norm of embedding delta (embedding mode)",
    )

    parser.add_argument(
        "--lambda-reg", type=float, default=0.1,
        help="Weight for correction regularization",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07,
        help="Softmax temperature for scoring",
    )
    parser.add_argument(
        "--loss-type", type=str, default="mcq",
        choices=["mcq", "contrastive", "combined"],
        help="Loss type: mcq (original), contrastive (InfoNCE), combined (both)",
    )
    parser.add_argument(
        "--alpha-mcq", type=float, default=1.0,
        help="Weight for MCQ loss in combined mode",
    )
    parser.add_argument(
        "--alpha-anti", type=float, default=0.0,
        help="Weight for anti-contrastive loss on negated MCQ options. "
        "Uses known negative (image, negated_text) pairs to teach "
        "image-discriminative corrections for retrieval. 0 = disabled.",
    )
    parser.add_argument(
        "--anti-margin", type=float, default=0.1,
        help="Margin for anti-contrastive hinge loss. The known non-match "
        "must score this much below the K-th best off-diagonal image.",
    )
    parser.add_argument(
        "--anti-topk", type=int, default=10,
        help="K for anti-contrastive hinge loss. Diagonal must rank below "
        "the K-th best off-diagonal. Higher K = harder constraint.",
    )
    parser.add_argument(
        "--anti-mode", type=str, default="hinge",
        choices=["hinge", "expanded"],
        help="Anti-contrastive mode: 'hinge' for separate hinge loss, "
        "'expanded' to put negated texts in the InfoNCE denominator.",
    )

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Limit training samples (for debugging)",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Enable mixed precision training (bf16 autocast, no GradScaler needed)",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Number of gradient accumulation steps before optimizer update",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help="EMA decay rate (0 = disabled, typical: 0.999)",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Label smoothing for cross-entropy losses (0.0 = off)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop if val metric doesn't improve for N epochs (0 = disabled)",
    )

    parser.add_argument(
        "--output-dir", type=str, default="results/score_corrector",
        help="Output directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save-every", type=int, default=1,
        help="Save checkpoint every N epochs",
    )

    parser.add_argument(
        "--use-wandb", action="store_true",
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="score-corrector",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity",
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs="+", default=None,
        help="W&B tags",
    )
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="Log step metrics every N batches",
    )
    parser.add_argument(
        "--exp-name", type=str, default=None,
        help="Experiment name",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)

    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = (
            f"sc_mc{args.max_correction}_lr{args.lambda_reg}"
            f"_t{args.temperature}_ctx{args.context_dim}_{timestamp}"
        )

    exp_dir = output_dir / args.exp_name
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["device"] = device
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("ScoreCorrector Training")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Device: {device}")
    print(f"Loss type: {args.loss_type}")
    if args.loss_type == "combined":
        print(f"Alpha MCQ: {args.alpha_mcq}")
    if args.alpha_anti > 0 or args.anti_mode == "expanded":
        print(f"Anti mode: {args.anti_mode}")
        if args.anti_mode == "hinge":
            print(f"Alpha anti-contrastive: {args.alpha_anti}")
            print(f"Anti margin: {args.anti_margin}, topk: {args.anti_topk}")
    print(f"Correction mode: {args.correction_mode}")
    if args.correction_mode == "margin_relative":
        print(f"Margin alpha: {args.margin_alpha}")
    if args.correction_mode == "multiplicative":
        print(f"Mult alpha: {args.mult_alpha}")
    if args.confidence_gate:
        print(f"Confidence gate: ON (rho={args.confidence_rho})")
    if args.correction_mode == "embedding":
        print(f"Embed alpha: {args.embed_alpha}")
    print(f"Layers: {args.layers}")
    print(f"Context dim: {args.context_dim}")
    print(f"Max correction: {args.max_correction}")
    print(f"Lambda reg: {args.lambda_reg}")
    print(f"Temperature: {args.temperature}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    if args.amp:
        print("AMP: bf16")
    if args.grad_accum_steps > 1:
        print(f"Gradient accumulation: {args.grad_accum_steps} steps")
    if args.ema_decay > 0:
        print(f"EMA decay: {args.ema_decay}")
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")
    if args.early_stopping_patience > 0:
        print(f"Early stopping patience: {args.early_stopping_patience}")
    print("=" * 60)

    use_wandb_logging = args.use_wandb and WANDB_AVAILABLE
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Continuing without wandb.")
            use_wandb_logging = False
        else:
            tags = args.wandb_tags or []
            tags.append("score-corrector")
            if args.loss_type != "mcq":
                tags.append(args.loss_type)

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.exp_name,
                config=config,
                tags=tags,
                save_code=True,
            )
            print(f"W&B run: {wandb.run.url}")

    print("\nCreating model...")
    model = ScoreCorrector(
        embed_dim=args.embed_dim,
        context_dim=args.context_dim,
        max_correction=args.max_correction,
        dropout=args.dropout,
        selected_layers=args.layers,
        use_gate=args.use_gate,
        correction_mode=args.correction_mode,
        margin_alpha=args.margin_alpha,
        mult_alpha=args.mult_alpha,
        confidence_gate=args.confidence_gate,
        confidence_rho=args.confidence_rho,
        embed_alpha=args.embed_alpha,
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Config: {model.get_model_info()}")

    loss_fn = ScoreCorrectorLoss(
        lambda_reg=args.lambda_reg,
        temperature=args.temperature,
        loss_type=args.loss_type,
        alpha_mcq=args.alpha_mcq,
        alpha_anti=args.alpha_anti,
        anti_margin=args.anti_margin,
        anti_topk=args.anti_topk,
        label_smoothing=args.label_smoothing,
        anti_mode=args.anti_mode,
    )

    print("\nLoading training data...")
    use_mmap = (Path(args.data_dir) / "image_emb.npy").exists()
    train_dataset = NegBenchDataset(
        data_dir=args.data_dir,
        layers=args.layers,
        load_to_memory=not use_mmap,
        shuffle_mcq=True,
    )
    if args.max_train_samples is not None:
        n_samples = min(args.max_train_samples, len(train_dataset))
        print(f"Limiting to {n_samples} samples")
        train_dataset = Subset(train_dataset, range(n_samples))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if args.val_dir:
        print("Loading validation data...")
        val_use_mmap = (Path(args.val_dir) / "image_emb.npy").exists()
        val_dataset = NegBenchDataset(
            data_dir=args.val_dir,
            layers=args.layers,
            load_to_memory=not val_use_mmap,
            shuffle_mcq=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    if use_wandb_logging:
        wandb.config.update({
            "model_params": model.count_parameters(),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if args.val_dir else 0,
            "total_steps": len(train_loader) * args.epochs,
        }, allow_val_change=True)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - args.warmup_steps, 1)
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    ema_params = None
    if args.ema_decay > 0:
        ema_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        print(f"EMA initialized with decay={args.ema_decay}")

    start_epoch = 0
    best_val_metric = 0.0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        start_epoch, loaded_metrics, loaded_ema = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        if loaded_ema is not None and ema_params is not None:
            for name in ema_params:
                if name in loaded_ema:
                    ema_params[name].copy_(loaded_ema[name].to(device))
            print("EMA state restored from checkpoint.")
        if args.loss_type == "contrastive":
            best_val_metric = (
                loaded_metrics.get("contrastive_acc_i2t", 0.0)
                + loaded_metrics.get("contrastive_acc_t2i", 0.0)
            ) / 2
        else:
            best_val_metric = loaded_metrics.get("mcq_accuracy", 0.0)
        print(f"Resumed from epoch {start_epoch}, best metric: {best_val_metric:.4f}")

    if args.loss_type != "mcq":
        print("\nRunning sanity check (score_contrastive vs score_pairwise)...")
        sanity_check_contrastive(model, train_loader, device)

    history = {"train": [], "val": [], "best_epoch": -1, "best_val_metric": 0.0}
    global_step = 0
    epochs_without_improvement = 0

    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        train_metrics = train_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_type=args.loss_type,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            epoch=epoch,
            global_step=global_step,
            use_wandb=use_wandb_logging,
            use_amp=args.amp,
            grad_accum_steps=args.grad_accum_steps,
            ema_params=ema_params,
            ema_decay=args.ema_decay,
            alpha_anti=args.alpha_anti,
            anti_mode=args.anti_mode,
        )
        global_step = train_metrics.pop("global_step", global_step)

        print(f"\nTrain: loss={train_metrics['loss_total']:.4f}")
        if "mcq_accuracy" in train_metrics:
            print(f"  mcq_acc={train_metrics['mcq_accuracy']:.4f}, "
                  f"baseline_acc={train_metrics['baseline_accuracy']:.4f}")
        if "loss_mcq" in train_metrics:
            print(f"  L_mcq={train_metrics['loss_mcq']:.4f}")
        if "loss_contrastive" in train_metrics:
            print(f"  L_contrastive={train_metrics['loss_contrastive']:.4f}")
        if "contrastive_acc_i2t" in train_metrics:
            print(f"  contrastive_acc: i2t={train_metrics['contrastive_acc_i2t']:.4f}, "
                  f"t2i={train_metrics['contrastive_acc_t2i']:.4f}")
        if "loss_anti" in train_metrics:
            print(f"  L_anti={train_metrics['loss_anti']:.4f}, "
                  f"anti_acc={train_metrics['anti_accuracy']:.4f}, "
                  f"violations={train_metrics.get('anti_violations', 0):.4f}, "
                  f"diag_rank={train_metrics.get('anti_mean_diag_rank', 0):.1f}")
        if "neg_above_pos" in train_metrics:
            print(f"  neg_above_pos={train_metrics['neg_above_pos']:.4f}")
        print(f"  L_reg={train_metrics['loss_reg']:.6f}, "
              f"mean|corr|={train_metrics['mean_abs_correction']:.4f}")

        history["train"].append(train_metrics)

        val_metrics = None
        if val_loader is not None:
            original_params = None
            if ema_params is not None:
                original_params = {}
                for name, param in model.named_parameters():
                    if name in ema_params:
                        original_params[name] = param.data.clone()
                        param.data.copy_(ema_params[name])
                print("  (evaluating with EMA weights)")

            val_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=val_loader,
                device=device,
                loss_type=args.loss_type,
                anti_mode=args.anti_mode,
            )

            print(f"\nVal: loss={val_metrics['loss_total']:.4f}")
            if "mcq_accuracy" in val_metrics:
                print(f"  mcq_acc={val_metrics['mcq_accuracy']:.4f}, "
                      f"baseline_acc={val_metrics['baseline_accuracy']:.4f}")
            if "contrastive_acc_i2t" in val_metrics:
                print(f"  contrastive_acc: i2t={val_metrics['contrastive_acc_i2t']:.4f}, "
                      f"t2i={val_metrics['contrastive_acc_t2i']:.4f}")
            if "neg_above_pos" in val_metrics:
                print(f"  neg_above_pos={val_metrics['neg_above_pos']:.4f}")
            print(f"  mean|corr|={val_metrics['mean_abs_correction']:.4f}")

            history["val"].append(val_metrics)

            if args.loss_type == "mcq" or args.loss_type == "combined":
                current_metric = val_metrics["mcq_accuracy"]
            else:
                current_metric = (
                    val_metrics["contrastive_acc_i2t"]
                    + val_metrics["contrastive_acc_t2i"]
                ) / 2

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                history["best_epoch"] = epoch
                history["best_val_metric"] = best_val_metric
                epochs_without_improvement = 0

                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    str(checkpoint_dir / "best_model.pt"), config,
                    ema_params=ema_params,
                )
                print("  -> New best model saved!")
            else:
                epochs_without_improvement += 1

            if original_params is not None:
                for name, param in model.named_parameters():
                    if name in original_params:
                        param.data.copy_(original_params[name])

        if use_wandb_logging:
            log_dict = {"epoch": epoch + 1}
            for k, v in train_metrics.items():
                log_dict[f"train/{k}"] = v
            if val_metrics:
                for k, v in val_metrics.items():
                    log_dict[f"val/{k}"] = v
                log_dict["best_val_metric"] = best_val_metric
            wandb.log(log_dict, step=global_step)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics if val_metrics else train_metrics,
                str(checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt"),
                config,
                ema_params=ema_params,
            )

        if (
            args.early_stopping_patience > 0
            and val_loader is not None
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"\nEarly stopping: no improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best val metric: {history['best_val_metric']:.4f}")
    print(f"Best epoch: {history['best_epoch'] + 1}")
    print(f"Outputs: {exp_dir}")

    if use_wandb_logging:
        wandb.run.summary["best_val_metric"] = history["best_val_metric"]
        wandb.run.summary["best_epoch"] = history["best_epoch"] + 1
        wandb.finish()


if __name__ == "__main__":
    main()
