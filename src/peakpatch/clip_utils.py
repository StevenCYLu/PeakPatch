"""CLIP utilities for on-the-fly token sequence extraction."""

from typing import Dict, List

import torch


def _get_text_encoder(model):
    """Get text encoder component from CLIP or CustomTextCLIP (SigLIP)."""
    if hasattr(model, "text") and hasattr(model.text, "transformer"):
        return model.text
    return model


def _is_custom_text_clip(model):
    """Check if model uses CustomTextCLIP architecture (e.g., SigLIP)."""
    return hasattr(model, "text") and hasattr(model.text, "transformer")


def _apply_text_projection(text_enc, x):
    """Apply text projection (handles nn.Linear vs nn.Parameter)."""
    tp = getattr(text_enc, "text_projection", None)
    if tp is None:
        return x
    if isinstance(tp, torch.nn.Linear):
        return tp(x)
    return x @ tp


def get_eos_positions(token_ids, model):
    """Get EOS token positions for CLIP or SigLIP tokens."""
    if _is_custom_text_clip(model):
        return (token_ids != 0).sum(dim=-1) - 1
    return token_ids.argmax(dim=-1)


@torch.no_grad()
def extract_token_sequences(
    token_ids: torch.Tensor,
    model,
    layers: List[int] = None,
) -> Dict[int, torch.Tensor]:
    """Run token_ids through frozen CLIP text encoder up to target layers."""
    if layers is None:
        layers = [6]

    max_layer = max(layers)
    text_enc = _get_text_encoder(model)
    cast_dtype = text_enc.transformer.get_cast_dtype()

    x = text_enc.token_embedding(token_ids.long()).to(cast_dtype)
    x = x + text_enc.positional_embedding[:x.size(1)].to(cast_dtype)

    attn_mask = text_enc.attn_mask if hasattr(text_enc, "attn_mask") else None

    result = {}
    for layer_idx, block in enumerate(text_enc.transformer.resblocks):
        x = block(x, attn_mask=attn_mask)
        current_layer = layer_idx + 1

        if current_layer in layers:
            result[current_layer] = text_enc.ln_final(x).float()

        if current_layer >= max_layer:
            break

    return result


@torch.no_grad()
def clip_text_forward_combined(
    token_ids: torch.Tensor,
    model,
    target_layer: int,
    anchor_layer: int,
    final_layer: int = None,
) -> Dict[str, torch.Tensor]:
    """Single CLIP forward pass returning anchor/target token seqs, final-layer CLS, and padding mask."""
    text_enc = _get_text_encoder(model)
    cast_dtype = text_enc.transformer.get_cast_dtype()

    if final_layer is None:
        final_layer = len(text_enc.transformer.resblocks)

    x = text_enc.token_embedding(token_ids.long()).to(cast_dtype)
    x = x + text_enc.positional_embedding[:x.size(1)].to(cast_dtype)
    attn_mask = text_enc.attn_mask if hasattr(text_enc, "attn_mask") else None

    eos_pos = get_eos_positions(token_ids, model)
    B = token_ids.shape[0]
    batch_idx = torch.arange(B, device=token_ids.device)

    needed = {anchor_layer, target_layer, final_layer}
    result = {}

    for layer_idx, block in enumerate(text_enc.transformer.resblocks):
        x = block(x, attn_mask=attn_mask)
        current_layer = layer_idx + 1

        if current_layer == anchor_layer:
            result["H_anchor"] = text_enc.ln_final(x).float()

        if current_layer == target_layer:
            result["H_target"] = text_enc.ln_final(x).float()

        if current_layer == final_layer:
            eos_features = x[batch_idx, eos_pos]
            eos_normed = text_enc.ln_final(eos_features)
            eos_projected = _apply_text_projection(text_enc, eos_normed)
            result["text_cls_L12"] = torch.nn.functional.normalize(
                eos_projected.float(), dim=-1)

        if current_layer >= final_layer:
            break

    result["padding_mask"] = compute_padding_mask(token_ids, model)
    return result


def compute_padding_mask(token_ids: torch.Tensor, model=None) -> torch.Tensor:
    """Compute padding mask from token IDs."""
    if model is not None and _is_custom_text_clip(model):
        return token_ids != 0

    eos_pos = token_ids.argmax(dim=-1)
    seq_len = token_ids.shape[1]
    positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
    mask = positions <= eos_pos.unsqueeze(1)
    return mask
