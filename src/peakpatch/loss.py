"""Loss functions for ScoreCorrector and joint EC+SC training."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreCorrectorLoss(nn.Module):
    """Loss module for ScoreCorrector training."""

    def __init__(
        self,
        lambda_reg: float = 0.1,
        temperature: float = 0.07,
        loss_type: str = "mcq",
        alpha_mcq: float = 1.0,
        alpha_anti: float = 0.0,
        anti_margin: float = 0.1,
        anti_topk: int = 10,
        label_smoothing: float = 0.0,
        anti_mode: str = "hinge",
    ):
        super().__init__()
        if loss_type not in ("mcq", "contrastive", "combined"):
            raise ValueError(f"Unknown loss_type: {loss_type}")
        if anti_mode not in ("hinge", "expanded"):
            raise ValueError(f"Unknown anti_mode: {anti_mode}")
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.loss_type = loss_type
        self.alpha_mcq = alpha_mcq
        self.alpha_anti = alpha_anti
        self.anti_margin = anti_margin
        self.anti_topk = anti_topk
        self.label_smoothing = label_smoothing
        self.anti_mode = anti_mode

    def forward(
        self,
        mcq_scores: torch.Tensor,
        mcq_corrections: torch.Tensor,
        mcq_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MCQ loss (unchanged for backward compat)."""
        logits = mcq_scores / self.temperature
        L_mcq = F.cross_entropy(logits, mcq_labels, label_smoothing=self.label_smoothing)

        L_reg = (mcq_corrections ** 2).mean()

        total = L_mcq + self.lambda_reg * L_reg

        return total, {
            "loss_total": total.item(),
            "loss_mcq": L_mcq.item(),
            "loss_reg": L_reg.item(),
            "mean_abs_correction": mcq_corrections.abs().mean().item(),
            "max_abs_correction": mcq_corrections.abs().max().item(),
        }

    def compute_contrastive_loss(
        self,
        scores: torch.Tensor,
        corrections: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute symmetric InfoNCE contrastive loss."""
        B = scores.shape[0]
        labels = torch.arange(B, device=scores.device)

        logits = scores / self.temperature

        L_i2t = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        L_t2i = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
        L_contrastive = (L_i2t + L_t2i) / 2

        L_reg = (corrections ** 2).mean()

        total = L_contrastive + self.lambda_reg * L_reg

        with torch.no_grad():
            acc_i2t = (logits.argmax(dim=1) == labels).float().mean().item()
            acc_t2i = (logits.T.argmax(dim=1) == labels).float().mean().item()

        return total, {
            "loss_total": total.item(),
            "loss_contrastive": L_contrastive.item(),
            "loss_reg": L_reg.item(),
            "contrastive_acc_i2t": acc_i2t,
            "contrastive_acc_t2i": acc_t2i,
            "mean_abs_correction": corrections.abs().mean().item(),
            "max_abs_correction": corrections.abs().max().item(),
        }

    def compute_anti_contrastive_loss(
        self,
        neg_scores: torch.Tensor,
        neg_corrections: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute hinge-based anti-contrastive loss on known negative pairs."""
        B = neg_scores.shape[0]
        K = min(self.anti_topk, B - 1)

        diag = torch.diag(neg_scores)

        mask = torch.ones(B, B, dtype=torch.bool, device=neg_scores.device)
        mask.fill_diagonal_(False)
        off_diag = neg_scores.masked_fill(~mask, float("-inf"))
        topk_vals = off_diag.topk(K, dim=0).values
        kth_best = topk_vals[-1]

        L_anti = F.relu(diag - kth_best + self.anti_margin).mean()

        with torch.no_grad():
            violations = (diag > kth_best - self.anti_margin).float().mean().item()
            argmax_per_col = neg_scores.argmax(dim=0)
            diag_indices = torch.arange(B, device=neg_scores.device)
            anti_acc = (argmax_per_col != diag_indices).float().mean().item()
            ranks = (neg_scores > diag.unsqueeze(0)).sum(dim=0).float()
            mean_diag_rank = ranks.mean().item()

        return L_anti, {
            "loss_anti": L_anti.item(),
            "anti_accuracy": anti_acc,
            "anti_violations": violations,
            "anti_mean_diag_rank": mean_diag_rank,
            "anti_mean_abs_correction": neg_corrections.abs().mean().item(),
        }

    def _forward_combined_expanded(
        self,
        pos_scores: torch.Tensor,
        pos_corrections: torch.Tensor,
        neg_scores: torch.Tensor,
        neg_corrections: torch.Tensor,
        mcq_scores: torch.Tensor,
        mcq_corrections: torch.Tensor,
        mcq_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with negated texts in the InfoNCE denominator."""
        B = pos_scores.shape[0]
        labels = torch.arange(B, device=pos_scores.device)

        expanded = torch.cat([pos_scores, neg_scores], dim=1)
        logits_i2t = expanded / self.temperature
        L_i2t = F.cross_entropy(logits_i2t, labels, label_smoothing=self.label_smoothing)

        logits_t2i = pos_scores.T / self.temperature
        L_t2i = F.cross_entropy(logits_t2i, labels, label_smoothing=self.label_smoothing)

        L_contrastive = (L_i2t + L_t2i) / 2

        mcq_logits = mcq_scores / self.temperature
        L_mcq = F.cross_entropy(mcq_logits, mcq_labels, label_smoothing=self.label_smoothing)

        all_corrections = torch.cat([
            pos_corrections.reshape(-1),
            neg_corrections.reshape(-1),
            mcq_corrections.reshape(-1),
        ])
        L_reg = (all_corrections ** 2).mean()

        total = L_contrastive + self.alpha_mcq * L_mcq + self.lambda_reg * L_reg

        with torch.no_grad():
            acc_i2t = (logits_i2t.argmax(dim=1) == labels).float().mean().item()
            acc_t2i = (logits_t2i.argmax(dim=1) == labels).float().mean().item()
            mcq_preds = mcq_scores.argmax(dim=1)
            mcq_acc = (mcq_preds == mcq_labels).float().mean().item()

            pos_diag = torch.diag(pos_scores)
            neg_diag = torch.diag(neg_scores)
            neg_above_pos = (neg_diag > pos_diag).float().mean().item()

        return total, {
            "loss_total": total.item(),
            "loss_contrastive": L_contrastive.item(),
            "loss_mcq": L_mcq.item(),
            "loss_reg": L_reg.item(),
            "contrastive_acc_i2t": acc_i2t,
            "contrastive_acc_t2i": acc_t2i,
            "mcq_accuracy": mcq_acc,
            "neg_above_pos": neg_above_pos,
            "mean_abs_correction": all_corrections.abs().mean().item(),
            "max_abs_correction": all_corrections.abs().max().item(),
        }

    def forward_combined(
        self,
        contrastive_scores: torch.Tensor,
        contrastive_corrections: torch.Tensor,
        mcq_scores: torch.Tensor,
        mcq_corrections: torch.Tensor,
        mcq_labels: torch.Tensor,
        anti_scores: torch.Tensor = None,
        anti_corrections: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined contrastive + MCQ + anti-contrastive loss."""
        if self.anti_mode == "expanded" and anti_scores is not None:
            return self._forward_combined_expanded(
                contrastive_scores, contrastive_corrections,
                anti_scores, anti_corrections,
                mcq_scores, mcq_corrections, mcq_labels,
            )

        B = contrastive_scores.shape[0]
        labels = torch.arange(B, device=contrastive_scores.device)

        logits = contrastive_scores / self.temperature

        L_i2t = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        L_t2i = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
        L_contrastive = (L_i2t + L_t2i) / 2

        mcq_logits = mcq_scores / self.temperature
        L_mcq = F.cross_entropy(mcq_logits, mcq_labels, label_smoothing=self.label_smoothing)

        correction_parts = [
            contrastive_corrections.reshape(-1),
            mcq_corrections.reshape(-1),
        ]
        if anti_corrections is not None:
            correction_parts.append(anti_corrections.reshape(-1))
        all_corrections = torch.cat(correction_parts)
        L_reg = (all_corrections ** 2).mean()

        total = L_contrastive + self.alpha_mcq * L_mcq + self.lambda_reg * L_reg

        with torch.no_grad():
            acc_i2t = (logits.argmax(dim=1) == labels).float().mean().item()
            acc_t2i = (logits.T.argmax(dim=1) == labels).float().mean().item()
            mcq_preds = mcq_scores.argmax(dim=1)
            mcq_acc = (mcq_preds == mcq_labels).float().mean().item()

        metrics = {
            "loss_total": total.item(),
            "loss_contrastive": L_contrastive.item(),
            "loss_mcq": L_mcq.item(),
            "loss_reg": L_reg.item(),
            "contrastive_acc_i2t": acc_i2t,
            "contrastive_acc_t2i": acc_t2i,
            "mcq_accuracy": mcq_acc,
            "mean_abs_correction": all_corrections.abs().mean().item(),
            "max_abs_correction": all_corrections.abs().max().item(),
        }

        if anti_scores is not None and self.alpha_anti > 0:
            L_anti, anti_metrics = self.compute_anti_contrastive_loss(
                anti_scores, anti_corrections,
            )
            total = total + self.alpha_anti * L_anti
            metrics["loss_total"] = total.item()
            metrics["loss_anti"] = anti_metrics["loss_anti"]
            metrics["anti_accuracy"] = anti_metrics["anti_accuracy"]
            metrics["anti_violations"] = anti_metrics["anti_violations"]
            metrics["anti_mean_diag_rank"] = anti_metrics["anti_mean_diag_rank"]

        return total, metrics

    def compute_mcq_accuracy(
        self,
        mcq_scores: torch.Tensor,
        mcq_labels: torch.Tensor,
    ) -> float:
        """Compute MCQ accuracy for a batch."""
        with torch.no_grad():
            predictions = mcq_scores.argmax(dim=1)
            accuracy = (predictions == mcq_labels).float().mean().item()
        return accuracy

    def compute_baseline_accuracy(
        self,
        mcq_layers: dict,
        image_emb: torch.Tensor,
        mcq_labels: torch.Tensor,
        anchor_layer: int = 12,
    ) -> float:
        """Compute baseline CLIP MCQ accuracy (no correction)."""
        with torch.no_grad():
            mcq_L12 = mcq_layers[anchor_layer]
            baseline_scores = torch.bmm(
                mcq_L12, image_emb.unsqueeze(-1)
            ).squeeze(-1)
            predictions = baseline_scores.argmax(dim=1)
            accuracy = (predictions == mcq_labels).float().mean().item()
        return accuracy


class JointLoss(nn.Module):
    """Loss module for joint EC+SC training."""

    def __init__(
        self,
        logit_scale: float = 100.0,
        sc_temperature: float = 0.07,
        lambda_sc: float = 1.0,
        lambda_reg: float = 0.1,
        lambda_reg_nonneg: float = 0.0,
    ):
        super().__init__()
        self.logit_scale = logit_scale
        self.sc_temperature = sc_temperature
        self.lambda_sc = lambda_sc
        self.lambda_reg = lambda_reg
        self.lambda_reg_nonneg = lambda_reg_nonneg

    def compute_ec_infonce(
        self,
        S_orig: torch.Tensor,
        S_neg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute EC InfoNCE loss on corrected similarity matrices."""
        B = S_orig.shape[0]
        labels = torch.arange(B, device=S_orig.device)

        logits_i2t = torch.cat([S_orig, S_neg], dim=1) * self.logit_scale
        L_i2t = F.cross_entropy(logits_i2t, labels)

        logits_t2i = S_orig.T * self.logit_scale
        L_t2i = F.cross_entropy(logits_t2i, labels)

        loss = (L_i2t + L_t2i) / 2

        with torch.no_grad():
            acc_i2t = (logits_i2t.argmax(dim=1) == labels).float().mean().item()
            acc_t2i = (logits_t2i.argmax(dim=1) == labels).float().mean().item()

        metrics = {
            "ec_loss": loss.item(),
            "ec_acc_i2t": acc_i2t,
            "ec_acc_t2i": acc_t2i,
        }
        return loss, metrics

    def compute_sc_ce(
        self,
        mcq_scores: torch.Tensor,
        mcq_corrections: torch.Tensor,
        mcq_labels: torch.Tensor,
        neg_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SC cross-entropy loss on MCQ scores."""
        logits = mcq_scores / self.sc_temperature
        L_ce = F.cross_entropy(logits, mcq_labels)

        if neg_mask is not None and self.lambda_reg_nonneg > 0:
            corr_sq = mcq_corrections ** 2
            neg_corr = corr_sq[neg_mask]
            nonneg_corr = corr_sq[~neg_mask]
            L_reg_neg = neg_corr.mean() if neg_corr.numel() > 0 else torch.tensor(0.0)
            L_reg_nonneg = nonneg_corr.mean() if nonneg_corr.numel() > 0 else torch.tensor(0.0)
            L_reg = self.lambda_reg * L_reg_neg + (self.lambda_reg + self.lambda_reg_nonneg) * L_reg_nonneg
            loss = L_ce + L_reg
        else:
            L_reg = (mcq_corrections ** 2).mean()
            loss = L_ce + self.lambda_reg * L_reg

        with torch.no_grad():
            acc = (mcq_scores.argmax(dim=1) == mcq_labels).float().mean().item()
            abs_corr = mcq_corrections.abs()
            corr_neg_val = abs_corr[neg_mask].mean().item() if neg_mask is not None and neg_mask.any() else 0.0
            corr_nonneg_val = abs_corr[~neg_mask].mean().item() if neg_mask is not None and (~neg_mask).any() else 0.0

        metrics = {
            "sc_loss": loss.item(),
            "sc_ce": L_ce.item(),
            "sc_reg": L_reg.item() if isinstance(L_reg, torch.Tensor) else L_reg,
            "sc_acc": acc,
            "sc_mean_correction": mcq_corrections.abs().mean().item(),
            "sc_corr_neg": corr_neg_val,
            "sc_corr_nonneg": corr_nonneg_val,
        }
        return loss, metrics

    def forward(
        self,
        S_orig: torch.Tensor,
        S_neg: torch.Tensor,
        mcq_scores: torch.Tensor,
        mcq_corrections: torch.Tensor,
        mcq_labels: torch.Tensor,
        neg_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined joint loss."""
        L_ec, ec_metrics = self.compute_ec_infonce(S_orig, S_neg)
        L_sc, sc_metrics = self.compute_sc_ce(
            mcq_scores, mcq_corrections, mcq_labels, neg_mask)

        total = L_ec + self.lambda_sc * L_sc

        metrics = {
            "loss_total": total.item(),
            **ec_metrics,
            **sc_metrics,
        }
        return total, metrics
