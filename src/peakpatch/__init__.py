"""PeakPatch: recovering negation from frozen CLIP intermediate features.

Two lightweight modules are trained jointly on top of a frozen CLIP backbone:

- ``EmbeddingCorrector`` (ECN, ~4.7M params) rewrites the final text embedding
  using token states read from an intermediate "peak" layer.
- ``ScoreCorrector`` (SCN, ~0.5M params) applies a bounded correction to the
  image-text score, using per-layer [EOS] features.

See the README for the checkpoints and the commands that reproduce the paper's
MCQ numbers.
"""

from .clip_utils import (
    clip_text_forward_combined,
    compute_padding_mask,
    extract_token_sequences,
    get_eos_positions,
)
from .dataset import (
    EmbeddingCorrectorDataset,
    MCQJointDataset,
    NegBenchDataset,
    NegcapJointDataset,
    collate_fn,
    mcq_collate_fn,
    negcap_collate_fn,
)
from .loss import JointLoss, ScoreCorrectorLoss
from .model import EmbeddingCorrector, ScoreCorrector

__all__ = [
    # models
    "EmbeddingCorrector",
    "ScoreCorrector",
    # losses
    "JointLoss",
    "ScoreCorrectorLoss",
    # datasets
    "EmbeddingCorrectorDataset",
    "MCQJointDataset",
    "NegBenchDataset",
    "NegcapJointDataset",
    "collate_fn",
    "mcq_collate_fn",
    "negcap_collate_fn",
    # CLIP helpers
    "clip_text_forward_combined",
    "compute_padding_mask",
    "extract_token_sequences",
    "get_eos_positions",
]
