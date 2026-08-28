"""Models for CLIP negation correction."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreCorrector(nn.Module):
    """Learned score-level corrector for CLIP cosine similarity."""

    VALID_MODES = ("fixed", "margin_relative", "cross_gate", "multiplicative", "two_head", "embedding")

    def __init__(
        self,
        embed_dim: int = 512,
        context_dim: int = 128,
        max_correction: float = 0.15,
        dropout: float = 0.1,
        selected_layers: List[int] = None,
        use_gate: bool = False,
        correction_mode: str = "fixed",
        margin_alpha: float = 0.5,
        mult_alpha: float = 0.3,
        confidence_gate: bool = False,
        confidence_rho: float = 1.0,
        embed_alpha: float = 0.1,
        use_token_features: bool = False,
        token_num_heads: int = 8,
    ):
        super().__init__()

        if selected_layers is None:
            selected_layers = [3, 8, 12]

        if correction_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid correction_mode={correction_mode!r}. "
                f"Must be one of {self.VALID_MODES}"
            )

        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.max_correction = max_correction
        self.selected_layers = selected_layers
        self.num_layers = len(selected_layers)
        self.correction_mode = correction_mode
        self.margin_alpha = margin_alpha
        self.mult_alpha = mult_alpha
        self.confidence_gate = confidence_gate
        self.confidence_rho = confidence_rho
        self.embed_alpha = embed_alpha
        self.use_token_features = use_token_features
        self.token_num_heads = token_num_heads

        if correction_mode == "embedding":
            if use_gate:
                import warnings
                warnings.warn("embedding mode ignores use_gate")
            if confidence_gate:
                import warnings
                warnings.warn("embedding mode ignores confidence_gate")
            self.use_gate = False

            self.embedding_corrector = nn.Sequential(
                nn.Linear(self.num_layers * embed_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, embed_dim),
            )
            self._init_weights()
            return

        if correction_mode == "cross_gate" and use_gate:
            import warnings
            warnings.warn(
                "cross_gate mode has its own gate; overriding use_gate=False"
            )
            use_gate = False
        self.use_gate = use_gate

        self.text_encoder = nn.Sequential(
            nn.Linear(self.num_layers * embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, context_dim),
        )

        self.cross_encoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, context_dim),
        )

        if use_token_features:
            self.token_query = nn.Parameter(
                torch.randn(1, 1, embed_dim) * 0.02)
            self.token_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=token_num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.token_attn_ln = nn.LayerNorm(embed_dim)
            self.token_proj = nn.Sequential(
                nn.Linear(embed_dim, context_dim),
                nn.LayerNorm(context_dim),
                nn.GELU(),
            )

        n_ctx = 3 if use_token_features else 2
        if correction_mode == "two_head":
            self.negation_head = nn.Linear(context_dim, 1)
            self.grounding_head = nn.Linear(context_dim, 1)
        else:
            head_input_dim = self.num_layers + n_ctx * context_dim
            self.correction_head = nn.Sequential(
                nn.Linear(head_input_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        if correction_mode == "cross_gate":
            self.cross_gate_head = nn.Linear(2 * context_dim, 1)

        if self.use_gate:
            self.gate_head = nn.Linear(context_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        if hasattr(self, "embedding_corrector"):
            for m in self.embedding_corrector:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            final_linear = self.embedding_corrector[-1]
            nn.init.normal_(final_linear.weight, std=0.001)
            nn.init.zeros_(final_linear.bias)
            return

        for module in [self.text_encoder, self.cross_encoder]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if hasattr(self, "correction_head"):
            for m in self.correction_head:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            final_linear = self.correction_head[-1]
            nn.init.normal_(final_linear.weight, std=0.001)
            nn.init.zeros_(final_linear.bias)

        if hasattr(self, "negation_head"):
            nn.init.normal_(self.negation_head.weight, std=0.001)
            nn.init.zeros_(self.negation_head.bias)
        if hasattr(self, "grounding_head"):
            nn.init.normal_(self.grounding_head.weight, std=0.001)
            nn.init.zeros_(self.grounding_head.bias)

        if hasattr(self, "cross_gate_head"):
            nn.init.normal_(self.cross_gate_head.weight, std=0.001)
            nn.init.zeros_(self.cross_gate_head.bias)

        if self.use_gate:
            nn.init.normal_(self.gate_head.weight, std=0.02)
            nn.init.zeros_(self.gate_head.bias)

        if hasattr(self, "token_proj"):
            for m in self.token_proj:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _compute_token_ctx(
        self,
        token_sequences: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token context via learned query cross-attention."""
        B = token_sequences.shape[0]
        query = self.token_query.expand(B, -1, -1)
        key_padding_mask = ~padding_mask

        attn_out, _ = self.token_attn(
            query=query,
            key=token_sequences,
            value=token_sequences,
            key_padding_mask=key_padding_mask,
        )
        summary = self.token_attn_ln(attn_out.squeeze(1))
        return self.token_proj(summary)

    def _compute_text_ctx(
        self,
        text_layers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute text context vector from multi-layer features."""
        layer_feats = [text_layers[layer] for layer in self.selected_layers]
        concat = torch.cat(layer_feats, dim=-1)
        return self.text_encoder(concat)

    def _compute_cross_ctx(
        self,
        text_L12: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-modal context from elementwise product."""
        cross = text_L12 * image_emb
        return self.cross_encoder(cross)

    def _compute_sims(
        self,
        text_layers: Dict[int, torch.Tensor],
        image_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarities for each layer."""
        sims = []
        for layer in self.selected_layers:
            sim = (text_layers[layer] * image_emb).sum(dim=-1, keepdim=True)
            sims.append(sim)
        return torch.cat(sims, dim=-1)

    def _compute_confidence_kappa(
        self,
        baseline_sim: torch.Tensor,
    ) -> torch.Tensor:
        """Compute kurtosis-based confidence from baseline similarities."""
        mu = baseline_sim.mean(dim=-1, keepdim=True)
        sigma = baseline_sim.std(dim=-1, keepdim=True).clamp(min=1e-6)
        standardized = (baseline_sim - mu) / sigma
        kurtosis = (standardized ** 4).mean(dim=-1, keepdim=True)
        kappa = (kurtosis / 3.0) ** self.confidence_rho
        return kappa.clamp(min=0.1, max=10.0)

    def _apply_correction(
        self,
        baseline_sim: torch.Tensor,
        sims: torch.Tensor,
        text_ctx: torch.Tensor,
        cross_ctx: torch.Tensor,
        precomputed_kappa: Optional[torch.Tensor] = None,
        token_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute final scores by applying mode-specific correction."""
        mode = self.correction_mode

        if mode == "two_head":
            neg = torch.tanh(self.negation_head(text_ctx)).squeeze(-1)
            ground = torch.sigmoid(self.grounding_head(cross_ctx)).squeeze(-1)
            correction = neg * ground * self.max_correction
        else:
            ctx_parts = [sims, text_ctx, cross_ctx]
            if token_ctx is not None:
                ctx_parts.append(token_ctx)
            features = torch.cat(ctx_parts, dim=-1)
            raw = self.correction_head(features).squeeze(-1)

            if mode == "fixed":
                correction = torch.tanh(raw) * self.max_correction

            elif mode == "margin_relative":
                local_spread = sims.max(dim=-1).values - sims.min(dim=-1).values
                correction = torch.tanh(raw) * self.margin_alpha * local_spread

            elif mode == "cross_gate":
                gate_input = torch.cat([text_ctx, cross_ctx], dim=-1)
                gate = torch.sigmoid(self.cross_gate_head(gate_input)).squeeze(-1)
                correction = gate * torch.tanh(raw) * self.max_correction

            elif mode == "multiplicative":
                correction = baseline_sim * self.mult_alpha * torch.tanh(raw)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_head(text_ctx)).squeeze(-1)
            correction = gate * correction

        if self.confidence_gate:
            if precomputed_kappa is not None:
                correction = correction / precomputed_kappa
            elif baseline_sim.dim() >= 2:
                kappa = self._compute_confidence_kappa(baseline_sim)
                correction = correction / kappa

        return baseline_sim + correction, correction

    def _compute_embedding_delta(
        self,
        text_layers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute bounded text embedding residual from multi-layer features."""
        layer_feats = [text_layers[layer] for layer in self.selected_layers]
        concat = torch.cat(layer_feats, dim=-1)
        raw = self.embedding_corrector(concat)
        raw_norm = raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.tanh(raw_norm / self.embed_alpha) * self.embed_alpha / raw_norm
        return raw * scale

    def forward(
        self,
        text_layers: Dict[int, torch.Tensor],
        image_emb: torch.Tensor,
        token_sequences: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute corrected score for (image, text) pairs."""
        anchor_layer = max(self.selected_layers)
        text_L12 = text_layers[anchor_layer]

        if self.correction_mode == "embedding":
            delta = self._compute_embedding_delta(text_layers)
            txt_corrected = F.normalize(text_L12 + delta, dim=-1)
            score = (image_emb * txt_corrected).sum(dim=-1)
            baseline_sim = (text_L12 * image_emb).sum(dim=-1)
            correction = score - baseline_sim
            return score, correction

        baseline_sim = (text_L12 * image_emb).sum(dim=-1)
        text_ctx = self._compute_text_ctx(text_layers)
        cross_ctx = self._compute_cross_ctx(text_L12, image_emb)
        sims = self._compute_sims(text_layers, image_emb)

        token_ctx = None
        if self.use_token_features:
            token_ctx = self._compute_token_ctx(token_sequences, padding_mask)

        corrected_score, correction = self._apply_correction(
            baseline_sim, sims, text_ctx, cross_ctx, token_ctx=token_ctx
        )
        return corrected_score, correction

    def score_mcq(
        self,
        mcq_layers: Dict[int, torch.Tensor],
        image_emb: torch.Tensor,
        mcq_token_seqs: Optional[torch.Tensor] = None,
        mcq_padding_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score MCQ options (4 options per sample)."""
        num_options = mcq_layers[self.selected_layers[0]].shape[1]

        all_scores = []
        all_corrections = []

        for opt_idx in range(num_options):
            opt_layers = {
                layer: mcq_layers[layer][:, opt_idx, :]
                for layer in self.selected_layers
            }
            token_seq = None
            mask = None
            if mcq_token_seqs is not None:
                token_seq = mcq_token_seqs[:, opt_idx]
                mask = mcq_padding_masks[:, opt_idx]
            score, correction = self.forward(
                opt_layers, image_emb, token_seq, mask)
            all_scores.append(score)
            all_corrections.append(correction)

        scores = torch.stack(all_scores, dim=1)
        corrections = torch.stack(all_corrections, dim=1)
        return scores, corrections

    def score_contrastive(
        self,
        text_layers: Dict[int, torch.Tensor],
        image_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute full B x B corrected score matrix for contrastive training."""
        anchor_layer = max(self.selected_layers)
        text_L12 = text_layers[anchor_layer]

        if self.correction_mode == "embedding":
            delta = self._compute_embedding_delta(text_layers)
            txt_corrected = F.normalize(text_L12 + delta, dim=-1)
            scores = image_emb @ txt_corrected.T
            return scores, delta

        baseline_sim = image_emb @ text_L12.T
        text_ctx = self._compute_text_ctx(text_layers)

        B = image_emb.shape[0]

        sims_list = []
        for layer in self.selected_layers:
            s = image_emb @ text_layers[layer].T
            sims_list.append(s.unsqueeze(-1))
        sims = torch.cat(sims_list, dim=-1)

        cross_input = image_emb.unsqueeze(1) * text_L12.unsqueeze(0)
        cross_ctx = self.cross_encoder(cross_input)

        text_ctx_exp = text_ctx.unsqueeze(0).expand(B, -1, -1)

        scores, correction = self._apply_correction(
            baseline_sim, sims, text_ctx_exp, cross_ctx
        )
        return scores, correction

    def score_pairwise(
        self,
        text_layers: Dict[int, torch.Tensor],
        image_embs: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Compute pairwise score matrix for retrieval."""
        N_img = image_embs.shape[0]
        N_text = text_layers[self.selected_layers[0]].shape[0]
        device = image_embs.device
        anchor_layer = max(self.selected_layers)

        if self.correction_mode == "embedding":
            text_L12 = text_layers[anchor_layer]
            delta = self._compute_embedding_delta(text_layers)
            txt_corrected = F.normalize(text_L12 + delta, dim=-1)
            scores = image_embs @ txt_corrected.T
            return scores

        text_ctx = self._compute_text_ctx(text_layers)
        text_L12 = text_layers[anchor_layer]
        baseline_sims = image_embs @ text_L12.T

        layer_sims = {}
        for layer in self.selected_layers:
            layer_sims[layer] = image_embs @ text_layers[layer].T

        full_kappa = None
        if self.confidence_gate:
            full_kappa = self._compute_confidence_kappa(baseline_sims)

        scores = torch.zeros(N_img, N_text, device=device)

        for i_start in range(0, N_img, batch_size):
            i_end = min(i_start + batch_size, N_img)
            B_img = i_end - i_start

            img_batch = image_embs[i_start:i_end]

            for j_start in range(0, N_text, batch_size):
                j_end = min(j_start + batch_size, N_text)
                B_text = j_end - j_start

                img_exp = img_batch.unsqueeze(1).expand(-1, B_text, -1)
                text_L12_exp = text_L12[j_start:j_end].unsqueeze(0).expand(B_img, -1, -1)

                cross_ctx = self._compute_cross_ctx(text_L12_exp, img_exp)
                text_ctx_exp = text_ctx[j_start:j_end].unsqueeze(0).expand(B_img, -1, -1)

                sims_list = []
                for layer in self.selected_layers:
                    s = layer_sims[layer][i_start:i_end, j_start:j_end]
                    sims_list.append(s.unsqueeze(-1))
                sims = torch.cat(sims_list, dim=-1)

                baseline_block = baseline_sims[i_start:i_end, j_start:j_end]

                block_kappa = None
                if full_kappa is not None:
                    block_kappa = full_kappa[i_start:i_end]

                block_scores, _ = self._apply_correction(
                    baseline_block, sims, text_ctx_exp, cross_ctx,
                    precomputed_kappa=block_kappa,
                )
                scores[i_start:i_end, j_start:j_end] = block_scores

        return scores

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        """Return model configuration info."""
        info = {
            "embed_dim": self.embed_dim,
            "context_dim": self.context_dim,
            "max_correction": self.max_correction,
            "selected_layers": self.selected_layers,
            "use_gate": self.use_gate,
            "correction_mode": self.correction_mode,
            "confidence_gate": self.confidence_gate,
            "use_token_features": self.use_token_features,
            "num_parameters": self.count_parameters(),
        }
        if self.correction_mode == "margin_relative":
            info["margin_alpha"] = self.margin_alpha
        if self.correction_mode == "multiplicative":
            info["mult_alpha"] = self.mult_alpha
        if self.correction_mode == "embedding":
            info["embed_alpha"] = self.embed_alpha
        if self.confidence_gate:
            info["confidence_rho"] = self.confidence_rho
        return info

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "ScoreCorrector":
        """Load a ScoreCorrector from a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})

        model = cls(
            embed_dim=config.get("embed_dim", 512),
            context_dim=config.get("context_dim", 128),
            max_correction=config.get("max_correction", 0.15),
            dropout=config.get("dropout", 0.1),
            selected_layers=config.get("layers", [3, 8, 12]),
            use_gate=config.get("use_gate", False),
            correction_mode=config.get("correction_mode", "fixed"),
            margin_alpha=config.get("margin_alpha", 0.5),
            mult_alpha=config.get("mult_alpha", 0.3),
            confidence_gate=config.get("confidence_gate", False),
            confidence_rho=config.get("confidence_rho", 1.0),
            embed_alpha=config.get("embed_alpha", 0.1),
            use_token_features=config.get("use_token_features", False),
            token_num_heads=config.get("token_num_heads", 8),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        return model


class EmbeddingCorrector(nn.Module):
    """Token-aware deviation predictor with learned query (no negation detector)."""

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        no_anchor: bool = False,
        cls_only: bool = False,
        no_pool: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.no_anchor = no_anchor
        self.cls_only = cls_only
        self.no_pool = no_pool

        if not cls_only:
            self.learned_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_ln = nn.LayerNorm(embed_dim)

        if cls_only:
            mlp_input_dim = embed_dim if no_anchor else 2 * embed_dim
        else:
            n_components = 2
            if not no_anchor:
                n_components += 1
            if not no_pool:
                n_components += 1
            mlp_input_dim = n_components * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.log_alpha = nn.Parameter(torch.log(torch.tensor(0.5)))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final_linear = self.mlp[-1]
        nn.init.normal_(final_linear.weight, std=0.001)
        nn.init.zeros_(final_linear.bias)

    def _get_cls_token(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the EOS (CLS) token from hidden states."""
        B = hidden_states.shape[0]
        eos_pos = padding_mask.float().sum(dim=-1).long() - 1
        eos_pos = eos_pos.clamp(min=0)
        batch_idx = torch.arange(B, device=hidden_states.device)
        return hidden_states[batch_idx, eos_pos]

    def _masked_mean_pool(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool hidden states over valid (non-padding) tokens."""
        mask_float = padding_mask.float().unsqueeze(-1)
        summed = (hidden_states * mask_float).sum(dim=1)
        counts = mask_float.sum(dim=1).clamp(min=1)
        return summed / counts

    def forward(
        self,
        H_anchor: torch.Tensor,
        H_target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict deviation vector using token-level features."""
        cls_target = self._get_cls_token(H_target, padding_mask)

        if self.cls_only:
            if self.no_anchor:
                features = cls_target
            else:
                cls_anchor = self._get_cls_token(H_anchor, padding_mask)
                features = torch.cat([cls_target, cls_anchor], dim=-1)
            return self.mlp(features)

        B = H_target.shape[0]

        query = self.learned_query.expand(B, -1, -1)
        key_padding_mask = ~padding_mask

        attn_out, _ = self.cross_attn(
            query=query,
            key=H_target,
            value=H_target,
            key_padding_mask=key_padding_mask,
        )
        attn_summary = self.attn_ln(attn_out.squeeze(1))

        parts = [cls_target, attn_summary]
        if not self.no_anchor:
            cls_anchor = self._get_cls_token(H_anchor, padding_mask)
            parts.append(cls_anchor)
        if not self.no_pool:
            pooled = self._masked_mean_pool(H_target, padding_mask)
            parts.append(pooled)
        features = torch.cat(parts, dim=-1)
        return self.mlp(features)

    def correct_embedding(
        self,
        text_cls: torch.Tensor,
        H_anchor: torch.Tensor,
        H_target: torch.Tensor,
        padding_mask: torch.Tensor,
        alpha: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply correction to a text embedding."""
        correction = self.forward(H_anchor, H_target, padding_mask)
        if alpha is None:
            alpha = torch.exp(self.log_alpha)
        corrected = text_cls + alpha * correction
        return F.normalize(corrected, dim=-1), correction

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        """Return model configuration info."""
        return {
            "model_type": "embedding_corrector",
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "no_anchor": self.no_anchor,
            "cls_only": self.cls_only,
            "no_pool": self.no_pool,
            "num_parameters": self.count_parameters(),
            "alpha": float(torch.exp(self.log_alpha).item()),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "EmbeddingCorrector":
        """Load an EmbeddingCorrector from a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        model_type = config.get("model_type", "embedding_corrector")
        # Older checkpoints store the model_type under its former name.
        if model_type == "token_aware_dp_v2":
            config["model_type"] = "embedding_corrector"

        model = cls(
            embed_dim=config.get("embed_dim", 512),
            hidden_dim=config.get("hidden_dim", 1024),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
            no_anchor=config.get("no_anchor", False),
            cls_only=config.get("cls_only", False),
            no_pool=config.get("no_pool", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        return model


