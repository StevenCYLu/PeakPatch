"""Dataset classes for pre-extracted NegBench features.

Two dataset types:
- NegBenchDataset: MCQ training data (image, text, 4 MCQ options)
- EmbeddingCorrectorDataset: Token-level DP training data (token IDs + CLS targets)

Expected directory layouts documented in each class.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NegBenchDataset(Dataset):
    """Dataset for pre-extracted NegBench features.

    Supports two loading modes:
    - load_to_memory=True: Load all data into RAM (fast training, high memory)
    - load_to_memory=False: Memory-map .npy files (instant startup, low memory)

    Expected directory layout:
        data_dir/
            text_layer_03.pt/.npy      # [N, 512] - Layer 3 text features
            text_layer_08.pt/.npy      # [N, 512] - Layer 8 text features
            text_layer_12.pt/.npy      # [N, 512] - Layer 12 text features
            image_emb.pt/.npy          # [N, 512] - Image embeddings
            mcq_layer_03.pt/.npy       # [N, 4, 512] - Layer 3 MCQ option features
            mcq_layer_08.pt/.npy       # [N, 4, 512] - Layer 8 MCQ option features
            mcq_layer_12.pt/.npy       # [N, 4, 512] - Layer 12 MCQ option features
            mcq_labels.pt/.npy         # [N] - Correct option indices (0-3)
            metadata.json              # Dataset metadata and sample info
    """

    def __init__(
        self,
        data_dir: str,
        layers: List[int] = None,
        load_to_memory: bool = False,
        shuffle_mcq: bool = True,
    ):
        """Initialize the dataset.

        Args:
            data_dir: Directory containing extracted features
            layers: Which layers to load (default: [3, 8, 12])
            load_to_memory: Whether to load all features into RAM.
            shuffle_mcq: Whether to randomly shuffle MCQ option order during
                training. This is CRITICAL for learning.
        """
        self.data_dir = Path(data_dir)
        self.layers = layers if layers is not None else [3, 8, 12]
        self.load_to_memory = load_to_memory
        self.use_mmap = not load_to_memory
        self.shuffle_mcq = shuffle_mcq

        if self.use_mmap:
            sample_npy = self.data_dir / "image_emb.npy"
            if not sample_npy.exists():
                raise FileNotFoundError(
                    f"Memory-mapped mode requires .npy files. "
                    f"Not found: {sample_npy}"
                )

        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self._load_features()
        self.num_samples = len(self.mcq_labels)

    def _load_features(self):
        """Load or memory-map feature files."""
        ext = ".npy" if self.use_mmap else ".pt"
        mode_str = "memory-mapped" if self.use_mmap else "in-memory"
        print(f"Loading features ({mode_str})...")

        self.text_features = {}
        for layer in self.layers:
            path = self.data_dir / f"text_layer_{layer:02d}{ext}"
            if path.exists():
                print(f"  Loading text_layer_{layer:02d}...", end=" ", flush=True)
                self.text_features[layer] = self._load_file(path)
                print("done")
            else:
                raise FileNotFoundError(f"Text features not found: {path}")

        image_path = self.data_dir / f"image_emb{ext}"
        if image_path.exists():
            print(f"  Loading image_emb...", end=" ", flush=True)
            self.image_emb = self._load_file(image_path)
            print("done")
        else:
            raise FileNotFoundError(f"Image embeddings not found: {image_path}")

        self.mcq_features = {}
        for layer in self.layers:
            path = self.data_dir / f"mcq_layer_{layer:02d}{ext}"
            if path.exists():
                print(f"  Loading mcq_layer_{layer:02d}...", end=" ", flush=True)
                self.mcq_features[layer] = self._load_file(path)
                print("done")
            else:
                raise FileNotFoundError(f"MCQ features not found: {path}")

        labels_path = self.data_dir / f"mcq_labels{ext}"
        if labels_path.exists():
            print(f"  Loading mcq_labels...", end=" ", flush=True)
            self.mcq_labels = self._load_file(labels_path)
            print("done")
        else:
            raise FileNotFoundError(f"MCQ labels not found: {labels_path}")

        print(f"Loaded NegBench dataset from {self.data_dir}")
        print(f"  Mode: {mode_str}")
        print(f"  Samples: {len(self.mcq_labels)}")
        print(f"  Layers: {self.layers}")
        if self.text_features:
            sample_shape = self.text_features[self.layers[0]].shape
            print(f"  Feature shape: {sample_shape}")

    def _load_file(self, path: Path) -> Union[torch.Tensor, np.ndarray]:
        """Load a feature file (either .pt or .npy with mmap)."""
        if self.use_mmap:
            return np.load(path, mmap_mode='r')
        else:
            return torch.load(path, map_location="cpu", weights_only=True)

    def __len__(self) -> int:
        return self.num_samples

    def _to_tensor(self, data) -> torch.Tensor:
        """Convert data to tensor (handles both numpy and torch)."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy())
        else:
            return data.clone()

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.

        Returns:
            Dict with:
                text_layers: {layer_idx: tensor [D]}
                image_emb: tensor [D]
                mcq_layers: {layer_idx: tensor [4, D]}
                mcq_label: int
        """
        original_label = int(self.mcq_labels[idx])

        mcq_layers_raw = {
            layer: self._to_tensor(self.mcq_features[layer][idx])
            for layer in self.layers
        }

        if self.shuffle_mcq:
            perm = torch.randperm(4)
            new_label = (perm == original_label).nonzero(as_tuple=True)[0].item()

            mcq_layers = {
                layer: mcq_layers_raw[layer][perm]
                for layer in self.layers
            }
        else:
            mcq_layers = mcq_layers_raw
            new_label = original_label

        return {
            "text_layers": {
                layer: self._to_tensor(self.text_features[layer][idx])
                for layer in self.layers
            },
            "image_emb": self._to_tensor(self.image_emb[idx]),
            "mcq_layers": mcq_layers,
            "mcq_label": new_label,
        }



class EmbeddingCorrectorDataset(Dataset):
    """Dataset for EmbeddingCorrector training.

    Loads pre-extracted token IDs (from extract_token_ids.py) and
    pre-extracted CLS features for target deviation computation.
    CLIP forward pass happens on-the-fly during training.

    Returns (token_ids, neg_L12_cls, target) where:
        - token_ids [77]: int32 BPE token IDs for CLIP forward pass
        - neg_L12_cls [512]: normalized negated L12 CLS embedding (projected)
        - target [512]: deviation = normalize(original_L12) - normalize(negated_L12)
    """

    def __init__(
        self,
        data_dir: str,
        max_samples: int = None,
    ):
        """Initialize EmbeddingCorrectorDataset.

        Args:
            data_dir: Directory containing negated_token_ids.pt and layer .pt files
            max_samples: Limit number of samples (None = all)
        """
        data_dir = Path(data_dir)

        # Load token IDs
        token_ids_path = data_dir / "negated_token_ids.pt"
        if not token_ids_path.exists():
            raise FileNotFoundError(
                f"Token IDs not found: {token_ids_path}. "
                f"Run scripts/extract_token_ids.py first."
            )
        self.token_ids = torch.load(token_ids_path, map_location="cpu", weights_only=True)

        # Load L12 CLS features for target computation
        self.neg_L12 = torch.load(
            data_dir / "negated_layer_12.pt", map_location="cpu", weights_only=True)
        self.orig_L12 = torch.load(
            data_dir / "original_layer_12.pt", map_location="cpu", weights_only=True)

        self.N = len(self.token_ids)
        if max_samples and max_samples < self.N:
            self.N = max_samples

        # Precompute target deltas
        orig_norm = F.normalize(self.orig_L12[:self.N], dim=-1)
        neg_norm = F.normalize(self.neg_L12[:self.N], dim=-1)
        self.target_delta = orig_norm - neg_norm

        print(f"Loaded EmbeddingCorrectorDataset: {self.N} samples")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """Get a single sample.

        Returns:
            Tuple of:
                token_ids: [77] int32 BPE token IDs
                neg_L12_cls: [512] normalized negated L12 CLS embedding
                target: [512] target deviation vector
        """
        token_ids = self.token_ids[idx]
        neg_L12_cls = F.normalize(self.neg_L12[idx], dim=-1)
        target = self.target_delta[idx]
        return token_ids, neg_L12_cls, target


class NegcapJointDataset(Dataset):
    """Dataset for negcap data used in joint EC+SC training.

    Loads pre-extracted negcap data: original/negated token IDs and L12 CLS
    embeddings plus image embeddings for InfoNCE contrastive loss.

    Expected directory layout:
        data_dir/
            image_emb.pt              # [N, 512]
            original_layer_12.pt      # [N, 512]
            negated_layer_12.pt       # [N, 512]
            original_token_ids.pt     # [N, 77]
            negated_token_ids.pt      # [N, 77]
    """

    def __init__(self, data_dir: str, max_samples: int = None):
        data_dir = Path(data_dir)

        self.image_emb = torch.load(
            data_dir / "image_emb.pt", map_location="cpu", weights_only=True)
        self.orig_L12 = torch.load(
            data_dir / "original_layer_12.pt", map_location="cpu", weights_only=True)
        self.neg_L12 = torch.load(
            data_dir / "negated_layer_12.pt", map_location="cpu", weights_only=True)
        self.orig_token_ids = torch.load(
            data_dir / "original_token_ids.pt", map_location="cpu", weights_only=True)
        self.neg_token_ids = torch.load(
            data_dir / "negated_token_ids.pt", map_location="cpu", weights_only=True)

        self.N = len(self.image_emb)
        if max_samples and max_samples < self.N:
            self.N = max_samples

        print(f"Loaded NegcapJointDataset: {self.N} samples from {data_dir}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            "image_emb": self.image_emb[idx],
            "orig_L12_cls": F.normalize(self.orig_L12[idx], dim=-1),
            "neg_L12_cls": F.normalize(self.neg_L12[idx], dim=-1),
            "orig_token_ids": self.orig_token_ids[idx],
            "neg_token_ids": self.neg_token_ids[idx],
        }


class MCQJointDataset(Dataset):
    """Dataset for MCQ data used in joint EC+SC training.

    Loads pre-extracted vanilla CLIP MCQ features plus token IDs for
    on-the-fly EC correction during training.

    Expected directory layout:
        data_dir/
            image_emb.pt              # [N, 512]
            mcq_layer_06.pt           # [N, 4, 512]
            mcq_layer_08.pt           # [N, 4, 512]
            mcq_layer_12.pt           # [N, 4, 512]
            mcq_token_ids.pt          # [N, 4, 77]
            mcq_labels.pt             # [N]
    """

    def __init__(
        self,
        data_dir: str,
        layers: List[int] = None,
        shuffle_mcq: bool = True,
    ):
        data_dir = Path(data_dir)
        self.layers = layers if layers is not None else [6, 8, 12]
        self.shuffle_mcq = shuffle_mcq

        self.image_emb = torch.load(
            data_dir / "image_emb.pt", map_location="cpu", weights_only=True)

        self.mcq_features = {}
        for layer in self.layers:
            path = data_dir / f"mcq_layer_{layer:02d}.pt"
            self.mcq_features[layer] = torch.load(
                path, map_location="cpu", weights_only=True)

        self.mcq_token_ids = torch.load(
            data_dir / "mcq_token_ids.pt", map_location="cpu", weights_only=True)
        self.mcq_labels = torch.load(
            data_dir / "mcq_labels.pt", map_location="cpu", weights_only=True)

        self.N = len(self.mcq_labels)
        print(f"Loaded MCQJointDataset: {self.N} samples, layers={self.layers}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        original_label = int(self.mcq_labels[idx])

        mcq_layers_raw = {
            layer: self.mcq_features[layer][idx].clone()
            for layer in self.layers
        }
        token_ids_raw = self.mcq_token_ids[idx].clone()  # [4, 77]

        if self.shuffle_mcq:
            perm = torch.randperm(4)
            new_label = (perm == original_label).nonzero(as_tuple=True)[0].item()
            mcq_layers = {
                layer: mcq_layers_raw[layer][perm] for layer in self.layers
            }
            token_ids = token_ids_raw[perm]
        else:
            mcq_layers = mcq_layers_raw
            token_ids = token_ids_raw
            new_label = original_label

        return {
            "image_emb": self.image_emb[idx],
            "mcq_layers": mcq_layers,
            "mcq_token_ids": token_ids,
            "mcq_label": new_label,
        }


def negcap_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for NegcapJointDataset."""
    return {
        "image_emb": torch.stack([b["image_emb"] for b in batch]),
        "orig_L12_cls": torch.stack([b["orig_L12_cls"] for b in batch]),
        "neg_L12_cls": torch.stack([b["neg_L12_cls"] for b in batch]),
        "orig_token_ids": torch.stack([b["orig_token_ids"] for b in batch]),
        "neg_token_ids": torch.stack([b["neg_token_ids"] for b in batch]),
    }


def mcq_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for MCQJointDataset."""
    layers = list(batch[0]["mcq_layers"].keys())
    return {
        "image_emb": torch.stack([b["image_emb"] for b in batch]),
        "mcq_layers": {
            layer: torch.stack([b["mcq_layers"][layer] for b in batch])
            for layer in layers
        },
        "mcq_token_ids": torch.stack([b["mcq_token_ids"] for b in batch]),
        "mcq_label": torch.tensor([b["mcq_label"] for b in batch]),
    }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for nested dict structure.

    Args:
        batch: List of sample dicts from NegBenchDataset

    Returns:
        Batched dict with stacked tensors
    """
    layers = list(batch[0]["text_layers"].keys())

    return {
        "text_layers": {
            layer: torch.stack([b["text_layers"][layer] for b in batch])
            for layer in layers
        },
        "image_emb": torch.stack([b["image_emb"] for b in batch]),
        "mcq_layers": {
            layer: torch.stack([b["mcq_layers"][layer] for b in batch])
            for layer in layers
        },
        "mcq_label": torch.tensor([b["mcq_label"] for b in batch]),
    }
