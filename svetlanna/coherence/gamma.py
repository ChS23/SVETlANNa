"""Reference implementation of partial coherence via the full mutual coherence matrix.

SpatialCoherence stores Γ(r1, r2) as a (H, W, H, W) tensor and provides
coherence_forward() for propagating Γ through linear optical elements.

This module is intended for cross-validation and testing only (O(N²) memory,
O(N³) per-element transform).  For production use, PartiallyCoherentWavefront
in wavefront_pc.py is far more efficient.
"""
from __future__ import annotations

import torch
from svetlanna.wavefront import Wavefront
from svetlanna.simulation_parameters import SimulationParameters


def _adjoint(g: torch.Tensor) -> torch.Tensor:
    """Hermitian adjoint of a mutual coherence tensor with shape (..., H, W, H, W).

    The adjoint swaps the two spatial coordinate pairs (r1 ↔ r2) and conjugates:
        [Γ†](r1, r2) = Γ*(r2, r1)
    which corresponds to swapping dims -4 ↔ -2 and -3 ↔ -1 then taking conj.
    """
    return g.conj().transpose(-4, -2).transpose(-3, -1)


class SpatialCoherence:
    """Mutual coherence function Γ(r1, r2) as a dense (H, W, H, W) tensor.

    Grid is limited to H*W ≤ 64² to keep memory O(N²) tractable.

    Parameters
    ----------
    data : torch.Tensor
        Coherence tensor of shape (..., H, W, H, W).
    sim_params : SimulationParameters
        Spatial simulation grid.
    """

    def __init__(
        self,
        data: torch.Tensor,
        sim_params: SimulationParameters,
        check_size: bool = True,
    ) -> None:
        H = len(sim_params.y)
        W = len(sim_params.x)
        if check_size and H * W > 64 * 64:
            raise ValueError(
                f"Grid {H}×{W} (H*W={H*W}) exceeds the 64²=4096 limit for "
                "SpatialCoherence. Use PartiallyCoherentWavefront instead."
            )
        if data.shape[-4] != H or data.shape[-3] != W or data.shape[-2] != H or data.shape[-1] != W:
            raise ValueError(
                f"data shape {tuple(data.shape)} does not match sim_params "
                f"grid ({H}, {W}, {H}, {W}) in the last 4 dimensions"
            )
        self.data = data
        self.sim_params = sim_params

    @property
    def intensity(self) -> torch.Tensor:
        """Diagonal of Γ: I(r) = Γ(r, r), shape (..., H, W)."""
        H = len(self.sim_params.y)
        W = len(self.sim_params.x)
        # Extract diagonal over the two spatial pairs
        g = self.data.reshape(*self.data.shape[:-4], H * W, H * W)
        diag = torch.diagonal(g, dim1=-2, dim2=-1)   # (..., H*W)
        return diag.reshape(*self.data.shape[:-4], H, W).real

    @classmethod
    def from_pcw(cls, pcw: "PartiallyCoherentWavefront") -> "SpatialCoherence":
        """Build a SpatialCoherence from a PartiallyCoherentWavefront (for testing)."""
        from .wavefront_pc import PartiallyCoherentWavefront  # noqa: F401  (type hint)
        return cls(pcw.coherence_matrix(), pcw.sim_params)

    def __repr__(self) -> str:
        H = len(self.sim_params.y)
        W = len(self.sim_params.x)
        return f"SpatialCoherence(grid=({H}, {W}), device={self.data.device})"


def coherence_forward(
    element: torch.nn.Module,
    gamma: SpatialCoherence,
) -> SpatialCoherence:
    """Propagate a mutual coherence function through a linear optical element.

    Computes Γ' = U Γ U† using the identity
        U Γ U† = (U (U Γ)†)†

    which requires only two applications of element.forward() without
    implementing the adjoint explicitly.  element.forward() acts on the trailing
    (y2, x2) axes of Γ; the leading (y1, x1) pair acts as a batch dimension.

    Parameters
    ----------
    element : nn.Module
        Any SVETlANNa optical element (FreeSpace, DiffractiveLayer, etc.).
    gamma : SpatialCoherence
        Input mutual coherence function.

    Returns
    -------
    SpatialCoherence
        Transformed coherence function Γ' = U Γ U†.

    Notes
    -----
    For zero-padded propagation methods (zpASM, zpRSC), the padding is applied
    to the trailing (y2, x2) axes only, so the leading (y1, x1) batch is unaffected.
    Verified in test_coherence_gamma.py.
    """
    g = gamma.data    # (..., H, W, H, W)

    # Step 1: apply U to trailing (y2, x2) axes — (y1, x1) is batch
    H = len(gamma.sim_params.y)
    W = len(gamma.sim_params.x)
    batch_shape = g.shape[:-4]
    g_flat = g.reshape(-1, H, W)                          # (batch*H*W, H, W)
    g1_flat = torch.Tensor(element.forward(Wavefront(g_flat)))
    g1 = g1_flat.reshape(*batch_shape, H, W, H, W)

    # Step 2: take Hermitian adjoint
    g1_adj = _adjoint(g1)                                  # (..., H, W, H, W)

    # Step 3: apply U again to the (now-swapped) trailing pair
    g1_adj_flat = g1_adj.reshape(-1, H, W)
    g2_flat = torch.Tensor(element.forward(Wavefront(g1_adj_flat)))
    g2 = g2_flat.reshape(*batch_shape, H, W, H, W)

    # Step 4: final adjoint gives Γ' = U Γ U†
    result = _adjoint(g2)

    return SpatialCoherence(result, gamma.sim_params, check_size=False)
