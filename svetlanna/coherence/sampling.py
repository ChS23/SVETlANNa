"""Stochastic field realization sampling for partially coherent sources.

Used when the optical system contains nonlinear elements (e.g., intensity-dependent
gain, detectors with saturation) where the coherent-mode propagation of
PartiallyCoherentWavefront.through() is no longer exact.

For purely linear systems prefer PartiallyCoherentWavefront.detect() instead.
"""
from __future__ import annotations

import torch
from svetlanna.wavefront import Wavefront
from svetlanna.simulation_parameters import SimulationParameters


def sample_realizations(
    pcw: "PartiallyCoherentWavefront",
    k: int,
    generator: torch.Generator | None = None,
) -> Wavefront:
    """Draw k stochastic field realizations from a PartiallyCoherentWavefront.

    Each realization is:
        ψ^(i) = Σ_n sqrt(λ_n) c_n^(i) φ_n   where  c_n ~ CN(0, 1) i.i.d.

    The ensemble average of |ψ^(i)|² converges to pcw.intensity as k→∞ with
    statistical error O(k^{-1/2}).

    Parameters
    ----------
    pcw : PartiallyCoherentWavefront
    k : int
        Number of realizations.
    generator : torch.Generator | None
        Optional RNG for reproducibility.

    Returns
    -------
    Wavefront
        Shape (k, H, W) (or (k, Nwl, H, W) for polychromatic sim_params).
        The leading 'realization' axis is NOT registered in any SimulationParameters;
        treat it as a batch dimension.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    mode_dim = pcw.sim_params_modes.index("mode")
    ndim = pcw.modes.ndim
    pos_dim = mode_dim if mode_dim >= 0 else ndim + mode_dim
    M = len(pcw.weights)

    # sqrt(λ_n) coefficients, shape (M,)
    sqrt_w = pcw.weights.sqrt()

    # CN(0,1) coefficients: c = (re + i*im) / sqrt(2), shape (k, M)
    real = torch.randn(k, M, device=pcw.modes.device, generator=generator)
    imag = torch.randn(k, M, device=pcw.modes.device, generator=generator)
    coeffs = (real + 1j * imag) * (sqrt_w / (2 ** 0.5))   # (k, M), complex

    # modes: (M, H, W) after moving mode to dim 0
    phi = torch.Tensor(pcw.modes).moveaxis(pos_dim, 0)     # (M, *spatial)
    spatial_shape = phi.shape[1:]                           # (H, W) or (Nwl, H, W)
    phi_2d = phi.reshape(M, -1)                            # (M, N)

    # ψ^(i) = Σ_n c_n^(i) φ_n   →   (k, N)
    psi_2d = coeffs @ phi_2d                               # (k, N)
    psi = psi_2d.reshape(k, *spatial_shape)

    return Wavefront(psi)


def incoherent_source(
    sim_params: SimulationParameters,
    intensity: torch.Tensor,
    k: int,
    generator: torch.Generator | None = None,
) -> Wavefront:
    """Stochastic realizations of a spatially incoherent (delta-correlated) source.

    Each realization is:
        ψ(r) = sqrt(I(r)) * exp(i * θ(r))   where  θ ~ Uniform(0, 2π) per pixel.

    Pixels are statistically independent (Γ(r1,r2) = I(r1) δ(r1-r2)).

    Parameters
    ----------
    sim_params : SimulationParameters
        Spatial simulation grid (used only for shape).
    intensity : torch.Tensor
        Shape (H, W). Mean intensity I(r) ≥ 0 at each pixel.
    k : int
        Number of realizations.
    generator : torch.Generator | None
        Optional RNG for reproducibility.

    Returns
    -------
    Wavefront
        Shape (k, H, W).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    I = intensity.clamp(min=0.0)
    amp = I.sqrt()                                          # (H, W)
    theta = torch.rand(k, *I.shape, device=I.device, generator=generator) * (2 * torch.pi)
    psi = amp.unsqueeze(0) * torch.exp(1j * theta)         # (k, H, W)
    return Wavefront(psi)


def detect_realizations(
    realizations: Wavefront,
    network: torch.nn.Module,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate the detected intensity from stochastic field realizations.

    Computes the (weighted) average of |S[ψ^(i)]|² over the realization batch:
        I_det ≈ (1/K) Σ_i w_i |S[ψ^(i)]|²

    The network S may contain NonlinearElement layers; this Monte-Carlo estimate
    is the only rigorous approach in that case.  Statistical error is O(K^{-1/2}).

    Parameters
    ----------
    realizations : Wavefront
        Shape (K, ..., H, W). Leading dim is the realization batch.
    network : nn.Module
        Optical network (may include nonlinear elements).
    weights : torch.Tensor | None
        Optional shape (K,) importance weights. Uniform if None.

    Returns
    -------
    torch.Tensor
        Estimated intensity, shape (..., H, W).
    """
    K = realizations.shape[0]
    out = network(realizations)                            # (K, ..., H, W)
    intensities = out.intensity                            # (K, ..., H, W)

    if weights is None:
        return intensities.mean(dim=0)
    else:
        w = weights.to(intensities.device)
        w = w / w.sum()
        # Broadcast weights over all trailing dims
        for _ in range(intensities.ndim - 1):
            w = w.unsqueeze(-1)
        return (w * intensities).sum(dim=0)
