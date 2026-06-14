"""Coherence source constructors for PartiallyCoherentWavefront.

All functions return a PartiallyCoherentWavefront whose modes are normalized
so that Σ_n λ_n = `power` (default 1.0) and each mode has unit L2 norm on the grid.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

from svetlanna.wavefront import Wavefront
from svetlanna.simulation_parameters import SimulationParameters

from .wavefront_pc import PartiallyCoherentWavefront, _make_sim_params_modes


# ── Gaussian Schell-model (analytic Starikov-Wolf decomposition) ───────────


def gaussian_schell_model(
    sim_params: SimulationParameters,
    waist_radius: float,
    coherence_width: float,
    power: float = 1.0,
    power_fraction: float = 0.99,
    max_modes: int | None = None,
) -> PartiallyCoherentWavefront:
    """Gaussian Schell-model (GSM) source via the Starikov-Wolf eigendecomposition.

    The 2-D cross-spectral density is separable:
        W(r1, r2) = exp(-a(x1²+x2²)/2) exp(-b(x1-x2)²/2)
                  × exp(-a(y1²+y2²)/2) exp(-b(y1-y2)²/2)

    where a = 1/w0² and b = 1/(2σc²).  The coherent modes are separable
    Hermite-Gaussian functions HG_{mn} with waist w_mode = sqrt(2/c).

    Parameters
    ----------
    sim_params : SimulationParameters
        Spatial simulation grid.
    waist_radius : float
        Gaussian beam 1/e² amplitude waist w0.
    coherence_width : float
        Transverse coherence radius σc.
    power : float
        Total optical power Σ λ_n (default 1.0).
    power_fraction : float
        Fraction of power to capture in the truncated mode set (default 0.99).
    max_modes : int | None
        Hard cap on M_1d (number of modes per dimension).

    Raises
    ------
    ValueError
        If coherence_width ≤ 0 (use sampling.incoherent_source for Δ-correlated fields).
    """
    if coherence_width <= 0:
        raise ValueError(
            "coherence_width must be > 0. "
            "For a fully incoherent (delta-correlated) source use "
            "svetlanna.coherence.sampling.incoherent_source()."
        )
    if power <= 0:
        raise ValueError("power must be positive")
    if not (0.0 < power_fraction < 1.0):
        raise ValueError("power_fraction must be in (0, 1)")

    # Starikov-Wolf parameters
    a = 1.0 / waist_radius ** 2
    b = 1.0 / (2.0 * coherence_width ** 2)
    c = math.sqrt(a ** 2 + 2.0 * a * b)
    q = b / (a + b + c)
    w_mode = math.sqrt(2.0 / c)          # HG mode waist in hermite_gauss convention

    # Number of 1-D modes: smallest M_1d with 1 - q^M_1d >= power_fraction
    if q < 1e-12:
        M_1d = 1                         # coherent limit: single mode
    else:
        M_1d = max(1, math.ceil(math.log(1.0 - power_fraction) / math.log(q)))
    if max_modes is not None:
        M_1d = min(M_1d, int(max_modes))

    M = M_1d * M_1d

    # 1-D eigenvalues λ_n = (1-q) q^n  (geometric series, normalized to 1)
    ns = torch.arange(M_1d, dtype=torch.float64)
    lam_1d = (1.0 - q) * q ** ns                           # (M_1d,)

    # 2-D eigenvalues from tensor product; rescale to target power
    lam_2d = (lam_1d.unsqueeze(1) * lam_1d.unsqueeze(0)).reshape(-1)  # (M,) row-major (m,n)
    lam_2d = lam_2d * (power / lam_2d.sum().item())

    # Grid spacing for discrete L2 norm (uniform grid assumed)
    Nx = len(sim_params.x)
    Ny = len(sim_params.y)
    dx = ((sim_params.x[-1] - sim_params.x[0]).abs() / max(Nx - 1, 1)).item()
    dy = ((sim_params.y[-1] - sim_params.y[0]).abs() / max(Ny - 1, 1)).item()

    # Generate HG modes and normalize each to unit L2 norm on the grid.
    # High-order modes (m+n ≥ ~25) can produce NaN on float32 grids because the
    # Hermite polynomial overflows while the Gaussian underflows to 0 at the grid
    # boundary, giving inf×0=NaN.  The analytic value there is 0 (Gaussian kills
    # everything far from center), so replacing NaN/inf with 0 is physically exact.
    mode_tensors: list[torch.Tensor] = []
    for m in range(M_1d):
        for n in range(M_1d):
            phi = Wavefront.hermite_gauss(sim_params, waist_radius=w_mode, m=m, n=n)
            phi_t = torch.Tensor(phi).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            norm_sq = (phi_t.abs().pow(2) * dx * dy).sum()
            phi_norm = phi_t / norm_sq.sqrt().clamp(min=1e-30)
            mode_tensors.append(phi_norm)

    # Stack along a new leading axis → (M, H, W)
    modes = Wavefront(torch.stack(mode_tensors, dim=0))
    weights = lam_2d.to(dtype=torch.float32, device=sim_params.device)
    sim_params_modes = _make_sim_params_modes(sim_params, M)

    return PartiallyCoherentWavefront(modes, weights, sim_params, sim_params_modes)


# ── Generic Γ → mode decomposition ────────────────────────────────────────


def modes_from_coherence_matrix(
    gamma: torch.Tensor,
    sim_params: SimulationParameters,
    power_fraction: float = 0.99,
) -> PartiallyCoherentWavefront:
    """Decompose a full mutual coherence matrix Γ into coherent modes.

    Performs the eigendecomposition Γ = Σ_n λ_n |φ_n><φ_n| via torch.linalg.eigh.

    Parameters
    ----------
    gamma : torch.Tensor
        Shape (H, W, H, W).  Must be Hermitian positive semi-definite.
    sim_params : SimulationParameters
        Spatial simulation grid matching the H×W of gamma.
    power_fraction : float
        Fraction of total power to retain in the truncated mode set.

    Notes
    -----
    Memory and compute are O(N²) and O(N³) where N = H*W.  Practical limit N ≤ 64².
    """
    if gamma.ndim != 4:
        raise ValueError(f"gamma must be 4-D (H,W,H,W), got {gamma.ndim}-D")
    H, W = gamma.shape[0], gamma.shape[1]
    if gamma.shape[2] != H or gamma.shape[3] != W:
        raise ValueError(f"gamma shape must be (H,W,H,W), got {tuple(gamma.shape)}")
    N = H * W
    if N > 64 * 64:
        warnings.warn(
            f"Grid {H}×{W} (N={N} > 64²=4096): eigh requires O(N³)={N**3:.2e} "
            "FLOPs. Consider using a coarser grid or schell_model().",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reshape and symmetrize for numerical stability
    g_mat = gamma.reshape(N, N)
    g_mat = (g_mat + g_mat.conj().mT) * 0.5

    # Eigendecomposition (eigh: Hermitian, ascending eigenvalues)
    vals, vecs = torch.linalg.eigh(g_mat)   # vals (N,), vecs (N, N) columns = eigenvectors

    # Clip negative numerical noise, then sort descending
    vals = vals.clamp(min=0.0)
    idx = torch.argsort(vals, descending=True)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Truncate to power_fraction
    total = vals.sum()
    cumsum = torch.cumsum(vals, dim=0)
    k = int((cumsum < power_fraction * total).sum().item()) + 1
    k = max(1, min(k, N))

    weights = vals[:k].to(torch.float32)
    # Each column of vecs is a mode; vecs[:, n] has shape (N,) → reshape to (H, W)
    modes_flat = vecs[:, :k].T.contiguous()          # (k, N)
    modes_spatial = modes_flat.reshape(k, H, W).to(torch.complex64)

    modes = Wavefront(modes_spatial)
    sim_params_modes = _make_sim_params_modes(sim_params, k)

    return PartiallyCoherentWavefront(modes, weights, sim_params, sim_params_modes)


# ── General Schell-model via coarse-grid decomposition ────────────────────


def schell_model(
    sim_params: SimulationParameters,
    intensity_profile: torch.Tensor,
    mu_profile: torch.Tensor,
    build_grid: int = 48,
    power_fraction: float = 0.99,
) -> PartiallyCoherentWavefront:
    """Decompose a general Schell-model source into coherent modes.

    The cross-spectral density is W(r1, r2) = sqrt(I(r1)) sqrt(I(r2)) μ(r1-r2)
    where μ is shift-invariant.  The decomposition is performed on a coarse
    build_grid×build_grid mesh, then modes are interpolated to the full grid.

    Parameters
    ----------
    sim_params : SimulationParameters
        Full spatial simulation grid.
    intensity_profile : torch.Tensor
        Shape (H, W). Intensity I(r) at each grid point.
    mu_profile : torch.Tensor
        Shape (H, W). Coherence degree μ(Δr) centered at pixel (H//2, W//2).
        μ(0) should equal 1 at the center.
    build_grid : int
        Side length of the coarse grid used for eigendecomposition.
    power_fraction : float
        Fraction of power to retain.
    """
    device = sim_params.device
    Hb = Wb = build_grid

    # Interpolate I and μ to coarse grid
    I_c = _interp2d(intensity_profile.float().to(device), (Hb, Wb))
    mu_c = _interp2d(mu_profile.float().to(device), (Hb, Wb))

    # Build Γ on coarse grid as outer product modulated by μ
    sqrt_I = I_c.clamp(min=0.0).sqrt()   # (Hb, Wb)

    # Evaluate μ(r1 - r2) for all coarse-grid pairs
    # mu_c center pixel at (Hb//2, Wb//2) corresponds to Δ = 0
    y_idx = torch.arange(Hb, device=device)
    x_idx = torch.arange(Wb, device=device)
    dY = (y_idx.unsqueeze(0) - y_idx.unsqueeze(1) + Hb // 2).clamp(0, Hb - 1)  # (Hb, Hb)
    dX = (x_idx.unsqueeze(0) - x_idx.unsqueeze(1) + Wb // 2).clamp(0, Wb - 1)  # (Wb, Wb)

    # mu_mat[(y1*Wb+x1), (y2*Wb+x2)] = mu_c[dY[y1,y2], dX[x1,x2]]
    dY_flat = dY.flatten()        # (Hb*Hb,)
    dX_flat = dX.flatten()        # (Wb*Wb,)
    # mu_c at all (dY, dX) combinations: (Hb*Hb, Wb*Wb)
    mu_mat_raw = mu_c[dY_flat[:, None], dX_flat[None, :]]  # (Hb*Hb, Wb*Wb)
    # Reshape and permute to (Hb, Wb, Hb, Wb)
    mu_4d = mu_mat_raw.reshape(Hb, Hb, Wb, Wb).permute(0, 2, 1, 3)  # (Hb, Wb, Hb, Wb)

    # Γ(r1, r2) = sqrt(I(r1)) * sqrt(I(r2)) * μ(r1-r2)
    outer = sqrt_I.unsqueeze(2).unsqueeze(3) * sqrt_I.unsqueeze(0).unsqueeze(1)  # (Hb, Wb, Hb, Wb)
    gamma_coarse = (outer * mu_4d).to(torch.complex64)

    # Create coarse sim_params for eigendecomposition
    sp_coarse = SimulationParameters(
        x=torch.linspace(sim_params.x[0].item(), sim_params.x[-1].item(), Wb, device=device),
        y=torch.linspace(sim_params.y[0].item(), sim_params.y[-1].item(), Hb, device=device),
        wavelength=sim_params.wavelength,
    )

    # Eigendecompose on coarse grid
    pcw_coarse = modes_from_coherence_matrix(gamma_coarse, sp_coarse, power_fraction)

    # Interpolate each coarse mode to the full grid and renormalize
    M = len(pcw_coarse.weights)
    H_full = len(sim_params.y)
    W_full = len(sim_params.x)
    dx = ((sim_params.x[-1] - sim_params.x[0]).abs() / max(W_full - 1, 1)).item()
    dy = ((sim_params.y[-1] - sim_params.y[0]).abs() / max(H_full - 1, 1)).item()

    coarse_phi = torch.Tensor(pcw_coarse.modes)  # (M, Hb, Wb)
    fine_list: list[torch.Tensor] = []
    for n in range(M):
        phi_r = coarse_phi[n].real.unsqueeze(0).unsqueeze(0)   # (1,1,Hb,Wb)
        phi_i = coarse_phi[n].imag.unsqueeze(0).unsqueeze(0)
        r_up = F.interpolate(phi_r, size=(H_full, W_full), mode="bilinear", align_corners=False)[0, 0]
        i_up = F.interpolate(phi_i, size=(H_full, W_full), mode="bilinear", align_corners=False)[0, 0]
        phi_up = torch.complex(r_up, i_up)
        norm = (phi_up.abs().pow(2) * dx * dy).sum().clamp(min=1e-60).sqrt()
        fine_list.append(phi_up / norm)

    modes = Wavefront(torch.stack(fine_list, dim=0).to(torch.complex64))
    sim_params_modes = _make_sim_params_modes(sim_params, M)

    return PartiallyCoherentWavefront(modes, pcw_coarse.weights, sim_params, sim_params_modes)


# ── Internal helpers ──────────────────────────────────────────────────────


def _interp2d(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly interpolate a (H, W) tensor to a new (H_out, W_out) size."""
    return F.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


# ── Filipovich 2024 discrete exponential coherence model ──────────────────


class FilipoviCoherenceSource:
    """Partially coherent MNIST source from the Filipovich 2024 coherence model.

    Cross-spectral density between source pixels m and m':
        J(m, m') = sqrt(I_m * I_m') * mu^‖r_m - r_m'‖_pixels

    The coherence matrix C_µ(m,m') = mu^‖r_m - r_m'‖ is image-independent;
    its eigenvectors {v_n} are precomputed once in ``__init__``.  Per-image
    coherent modes are  φ_n = sqrt(I) ⊙ v_n  (element-wise on the DONN grid).

    Parameters
    ----------
    mu : float
        Coherence parameter in [0, 1].  mu=1 → coherent; mu=0 → incoherent.
    src_shape : tuple[int, int]
        (H_src, W_src) of the SOURCE pixel grid (e.g. (28, 28) for MNIST).
        C_µ is built on this grid; eigenvectors are then upsampled to sim_params.
    sim_params : SimulationParameters
        DONN propagation grid (must match the intensity tensors passed to from_image).
    power_fraction : float
        Fraction of source power captured in the truncated mode set.
    max_modes : int | None
        Hard cap on the number of retained modes (useful for incoherent sources).
    device : torch.device | None
        Move weight/vector tensors to this device after construction.
    """

    def __init__(
        self,
        mu: float,
        src_shape: tuple[int, int],
        sim_params: SimulationParameters,
        power_fraction: float = 0.99,
        max_modes: int | None = None,
        device=None,
    ) -> None:
        H_src, W_src = src_shape
        N = H_src * W_src

        # --- build C_µ on the source pixel grid ---
        ys = torch.arange(H_src, dtype=torch.float32)
        xs = torch.arange(W_src, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gy.flatten(), gx.flatten()], dim=1)  # (N, 2)
        dist = torch.cdist(coords, coords)                          # (N, N)

        if mu == 0.0:
            C = torch.eye(N, dtype=torch.float64)
        elif mu == 1.0:
            C = torch.ones(N, N, dtype=torch.float64)
        else:
            C = (mu ** dist).to(torch.float64)

        # --- eigendecompose (eigh: Hermitian, returns ascending eigenvalues) ---
        vals, vecs = torch.linalg.eigh(C)   # vals (N,), vecs (N, N) columns
        vals = vals.clamp(min=0.0).flip(0)  # descending
        vecs = vecs.flip(1)                 # (N, N), columns now descending

        # --- truncate to power_fraction ---
        total = vals.sum()
        if total > 0:
            M = int((torch.cumsum(vals, 0) < power_fraction * total).sum()) + 1
        else:
            M = 1
        if max_modes is not None:
            M = min(M, int(max_modes))
        M = max(1, M)

        # --- upsample eigenvectors from src grid to DONN propagation grid ---
        H_sp = len(sim_params.y)
        W_sp = len(sim_params.x)
        vecs_top = vecs[:, :M].T.float()               # (M, N)
        vecs_2d  = vecs_top.reshape(M, 1, H_src, W_src)
        if (H_src, W_src) == (H_sp, W_sp):
            vecs_up = vecs_2d.squeeze(1)               # no interpolation needed
        else:
            vecs_up = F.interpolate(
                vecs_2d, size=(H_sp, W_sp), mode="bilinear", align_corners=False
            ).squeeze(1)                               # (M, H_sp, W_sp)

        self.weights:   torch.Tensor = vals[:M].float()   # (M,)
        self.vecs_up:   torch.Tensor = vecs_up            # (M, H_sp, W_sp)
        self.src_shape  = src_shape
        self.M          = M
        self.sim_params = sim_params
        self.sim_params_modes = _make_sim_params_modes(sim_params, M)

        if device is not None:
            self.to(device)

    # ── helpers ───────────────────────────────────────────────────────────────

    def to(self, device) -> "FilipoviCoherenceSource":
        """Move weight and vector tensors to device (in-place)."""
        self.weights  = self.weights.to(device)
        self.vecs_up  = self.vecs_up.to(device)
        return self

    # ── per-image source generation ───────────────────────────────────────────

    def from_image(self, intensity: torch.Tensor) -> "PartiallyCoherentWavefront":
        """Generate a PCW for a single MNIST input image.

        Parameters
        ----------
        intensity : torch.Tensor
            Shape (H_sp, W_sp) — intensity on the DONN propagation grid,
            non-negative real values.  Typically obtained by bilinearly
            resizing a 28×28 MNIST image to match sim_params.

        Returns
        -------
        PartiallyCoherentWavefront
            Modes (M, H_sp, W_sp) with weights from C_µ eigendecomposition.
        """
        sqrt_I = intensity.clamp(min=0.0).sqrt()                   # (H_sp, W_sp)
        modes_real = self.vecs_up * sqrt_I.unsqueeze(0)            # (M, H_sp, W_sp)
        modes = Wavefront(modes_real.to(torch.complex64))
        return PartiallyCoherentWavefront(
            modes, self.weights.clone(), self.sim_params, self.sim_params_modes
        )
