from typing import *
import torch, torch.nn as nn

class Elasticity(nn.Module):
    """Base class for elasticity models with a few helpers."""
    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.dim = dim

    @staticmethod
    def transpose(M: torch.Tensor) -> torch.Tensor:
        return M.transpose(1, 2)

    @staticmethod
    def svd(F: torch.Tensor):
        # Robust SVD for batched 3x3
        U, S, Vh = torch.linalg.svd(F, full_matrices=False)
        return U, S, Vh

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

class Plasticity(nn.Module):
    """Base class for plasticity models."""
    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.dim = dim

    @staticmethod
    def transpose(M: torch.Tensor) -> torch.Tensor:
        return M.transpose(1, 2)

    @staticmethod
    def svd(F: torch.Tensor):
        U, S, Vh = torch.linalg.svd(F, full_matrices=False)
        return U, S, Vh

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
