"""PeakPatch: recovering negation from frozen CLIP intermediate features."""

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
    "EmbeddingCorrector",
    "ScoreCorrector",
    "JointLoss",
    "ScoreCorrectorLoss",
    "EmbeddingCorrectorDataset",
    "MCQJointDataset",
    "NegBenchDataset",
    "NegcapJointDataset",
    "collate_fn",
    "mcq_collate_fn",
    "negcap_collate_fn",
    "clip_text_forward_combined",
    "compute_padding_mask",
    "extract_token_sequences",
    "get_eos_positions",
]
