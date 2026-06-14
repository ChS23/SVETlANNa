import torch
from svetlanna.wavefront import Wavefront
from svetlanna.simulation_parameters import SimulationParameters


def _weights_broadcast(weights: torch.Tensor, ndim: int, mode_dim: int) -> torch.Tensor:
    """Reshape (M,) weights to broadcast over a tensor of `ndim` dims with mode at `mode_dim`."""
    shape = [1] * ndim
    shape[mode_dim] = -1
    return weights.reshape(shape)


def _make_sim_params_modes(sim_params: SimulationParameters, M: int) -> SimulationParameters:
    """Extend sim_params with a 'mode' axis of size M appended last.

    Appending last makes 'mode' the outermost (most-leading) non-scalar axis in tensors,
    consistent with SVETlANNa's convention that later-inserted axes have smaller negative indices.
    """
    axes = {n: buf.clone() for n, buf in sim_params._buffers.items() if buf is not None}
    axes["mode"] = torch.arange(M, device=sim_params.device)
    return SimulationParameters(axes)


def _rebuild_sim_params_modes(sp_modes: SimulationParameters, new_M: int) -> SimulationParameters:
    """Return a copy of sp_modes with the 'mode' axis replaced by torch.arange(new_M)."""
    axes = {n: buf.clone() for n, buf in sp_modes._buffers.items() if buf is not None and n != "mode"}
    axes["mode"] = torch.arange(new_M, device=sp_modes.device)
    return SimulationParameters(axes)


class PartiallyCoherentWavefront:
    """Partially coherent optical field in coherent-mode (Mercer) representation.

    The mutual coherence function is decomposed as:
        Γ(r1, r2) = Σ_n λ_n φ_n*(r1) φ_n(r2)

    where φ_n are orthonormal coherent modes and λ_n ≥ 0 are their weights.
    Each mode propagates independently through any linear system, so a single
    batched forward pass propagates the full partially coherent beam.

    Parameters
    ----------
    modes : Wavefront
        Complex field amplitudes of the coherent modes. Shape (..., M, H, W)
        where the 'mode' axis is registered in sim_params_modes as the last-added
        (outermost) non-scalar axis.
    weights : torch.Tensor
        Eigenvalues λ_n ≥ 0, shape (M,). Sum = total optical power.
    sim_params : SimulationParameters
        Spatial simulation parameters used by optical elements (no 'mode' axis).
    sim_params_modes : SimulationParameters
        sim_params extended with 'mode' axis; used internally for axis indexing.
    """

    def __init__(
        self,
        modes: Wavefront,
        weights: torch.Tensor,
        sim_params: SimulationParameters,
        sim_params_modes: SimulationParameters,
    ) -> None:
        self._validate(modes, weights, sim_params, sim_params_modes)
        self.modes = modes
        self.weights = weights
        self.sim_params = sim_params
        self.sim_params_modes = sim_params_modes

    # ── validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(
        modes: Wavefront,
        weights: torch.Tensor,
        sim_params: SimulationParameters,
        sim_params_modes: SimulationParameters,
    ) -> None:
        if "mode" not in sim_params_modes:
            raise ValueError("'mode' axis must be registered in sim_params_modes")
        if "mode" in sim_params:
            raise ValueError("sim_params must not contain a 'mode' axis")
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {tuple(weights.shape)}")
        M = len(weights)
        if M == 0:
            raise ValueError("At least one mode is required")
        if not modes.is_complex():
            raise TypeError(f"modes must be complex-valued, got {modes.dtype}")
        mode_dim = sim_params_modes.index("mode")
        ndim = modes.ndim
        pos_dim = mode_dim if mode_dim >= 0 else ndim + mode_dim
        if pos_dim < 0 or pos_dim >= ndim:
            raise ValueError(
                f"mode_dim={mode_dim} is out of range for modes with {ndim} dimensions"
            )
        if modes.shape[mode_dim] != M:
            raise ValueError(
                f"modes.shape[{mode_dim}] = {modes.shape[mode_dim]} "
                f"does not match len(weights) = {M}"
            )
        if not weights.real.ge(0).all():
            raise ValueError("All eigenvalues (λ_n) must be non-negative")

    # ── observable quantities ─────────────────────────────────────────────────

    @property
    def intensity(self) -> torch.Tensor:
        """Incoherent superposition Σ_n λ_n |φ_n|²."""
        mode_dim = self.sim_params_modes.index("mode")
        w = _weights_broadcast(self.weights, self.modes.ndim, mode_dim)
        return (w * self.modes.intensity).sum(dim=mode_dim)

    @property
    def total_power(self) -> torch.Tensor:
        """Total optical power = Σ_n λ_n."""
        return self.weights.sum()

    def degree_of_coherence(
        self, idx1: tuple[int, int], idx2: tuple[int, int]
    ) -> torch.Tensor:
        """Complex degree of coherence μ(r1, r2) = Γ(r1,r2) / sqrt(I(r1) I(r2)).

        Parameters
        ----------
        idx1, idx2 : (y_index, x_index)
            Integer pixel indices for the two spatial points.
        """
        y1, x1 = idx1
        y2, x2 = idx2
        phi = torch.Tensor(self.modes)        # (..., M, H, W)
        phi1 = phi[..., y1, x1]               # (..., M)
        phi2 = phi[..., y2, x2]               # (..., M)
        w = self.weights                       # (M,)
        gamma12 = (w * phi1.conj() * phi2).sum(dim=-1)
        I1 = (w * phi1.abs().pow(2)).sum(dim=-1)
        I2 = (w * phi2.abs().pow(2)).sum(dim=-1)
        eps = torch.finfo(I1.real.dtype if I1.is_complex() else I1.dtype).tiny
        return gamma12 / (I1 * I2).clamp(min=0).sqrt().clamp(min=eps)

    def coherence_matrix(self, force: bool = False) -> torch.Tensor:
        """Full mutual coherence function Γ as a (H, W, H, W) tensor.

        For debug and cross-validation only. Raises if H*W > 64² unless force=True.
        """
        H, W = self.sim_params.axis_sizes(("y", "x"))
        if H * W > 64 * 64 and not force:
            raise RuntimeError(
                f"Grid {H}×{W} (H*W={H*W}) is too large for the full coherence matrix. "
                "Pass force=True to override (O(M·H²W²) memory)."
            )
        mode_dim = self.sim_params_modes.index("mode")
        ndim = self.modes.ndim
        pos_dim = mode_dim if mode_dim >= 0 else ndim + mode_dim
        # Move mode to front, flatten spatial, compute Γ = Σ_n λ_n φ_n* ⊗ φ_n
        phi = torch.Tensor(self.modes).moveaxis(pos_dim, 0)   # (M, ..., H, W)
        M = len(self.weights)
        phi_2d = phi.reshape(M, -1, H * W)                    # (M, batch, HW)
        w = self.weights[:, None, None]                        # (M, 1, 1)
        # Γ(r1, r2) = Σ_n λ_n φ_n*(r1) φ_n(r2)
        gamma_flat = torch.einsum("mbi,mbj->ij", w * phi_2d.conj(), phi_2d)  # (HW, HW)
        return gamma_flat.reshape(H, W, H, W)

    # ── propagation ───────────────────────────────────────────────────────────

    def through(self, network: torch.nn.Module) -> "PartiallyCoherentWavefront":
        """Propagate all modes through a linear optical network in one batched pass.

        Weights are unchanged — exact for any linear transformation.
        """
        out = Wavefront(network(self.modes))
        return PartiallyCoherentWavefront(out, self.weights, self.sim_params, self.sim_params_modes)

    def detect(self, network: torch.nn.Module) -> torch.Tensor:
        """Compute detected intensity I_det = Σ_n λ_n |S[φ_n]|².

        Uses a single batched forward pass. Gradients flow through all modes
        to network parameters, enabling DONN training under partial coherence.

        Parameters
        ----------
        network : nn.Module
            Linear optical network ending just before the detector.

        Returns
        -------
        torch.Tensor
            Intensity image with the mode axis summed out.
        """
        out = network(self.modes)
        mode_dim = self.sim_params_modes.index("mode")
        w = _weights_broadcast(self.weights, out.ndim, mode_dim)
        return (w * out.intensity).sum(dim=mode_dim)

    # ── truncation / orthogonalization ────────────────────────────────────────

    def truncate(self, power_fraction: float = 0.99) -> "PartiallyCoherentWavefront":
        """Return a new PCW keeping the fewest modes that capture power_fraction of power.

        Modes are sorted by λ_n descending before truncation.
        """
        sorted_idx = torch.argsort(self.weights, descending=True)
        sorted_w = self.weights[sorted_idx]
        cumsum = torch.cumsum(sorted_w, dim=0)
        k = int((cumsum < power_fraction * self.total_power).sum().item()) + 1
        k = max(1, min(k, len(self.weights)))

        keep = sorted_idx[:k].to(torch.int64)
        new_weights = self.weights[keep]
        mode_dim = self.sim_params_modes.index("mode")
        ndim = self.modes.ndim
        pos_dim = mode_dim if mode_dim >= 0 else ndim + mode_dim
        new_modes = Wavefront(torch.index_select(torch.Tensor(self.modes), pos_dim, keep))

        new_sp_modes = _rebuild_sim_params_modes(self.sim_params_modes, k)
        return PartiallyCoherentWavefront(new_modes, new_weights, self.sim_params, new_sp_modes)

    def reorthogonalize(self) -> "PartiallyCoherentWavefront":
        """Restore canonical Mercer form via SVD of sqrt(λ)·Φ.

        Needed after lossy elements (amplitude apertures, absorption) that destroy
        mode orthogonality. Unitary elements (phase masks, free-space propagation)
        preserve orthogonality, so this is a no-op for pure phase systems.
        """
        mode_dim = self.sim_params_modes.index("mode")
        ndim = self.modes.ndim
        pos_dim = mode_dim if mode_dim >= 0 else ndim + mode_dim
        M = len(self.weights)

        phi = torch.Tensor(self.modes)
        phi_perm = phi.moveaxis(pos_dim, 0)       # (M, *spatial_and_batch)
        trailing = phi_perm.shape[1:]
        phi_2d = phi_perm.reshape(M, -1)           # (M, N)

        sqrt_w = self.weights.sqrt().unsqueeze(-1)  # (M, 1)
        # SVD of (sqrt(λ)·Φ): captures the same Γ = Φ^† diag(λ) Φ
        _, S, Vh = torch.linalg.svd(sqrt_w * phi_2d, full_matrices=False)

        new_M = S.shape[0]
        new_weights = S ** 2
        new_phi_perm = Vh.reshape(new_M, *trailing)
        new_phi = new_phi_perm.moveaxis(0, pos_dim).to(phi.dtype)

        new_sp_modes = _rebuild_sim_params_modes(self.sim_params_modes, new_M)
        return PartiallyCoherentWavefront(
            Wavefront(new_phi), new_weights, self.sim_params, new_sp_modes
        )

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample_realizations(self, k: int, generator: torch.Generator | None = None) -> Wavefront:
        """Draw k stochastic realizations of the field.

        Each realization is ψ^(k) = Σ_n sqrt(λ_n) c_n^(k) φ_n where c_n ~ CN(0,1).
        The ensemble average of |ψ^(k)|² converges to intensity as k→∞.

        Returns a Wavefront with a leading 'realization' axis of size k.
        """
        from .sampling import sample_realizations as _sample
        return _sample(self, k, generator=generator)

    # ── device / dtype ────────────────────────────────────────────────────────

    def to(self, *args, **kwargs) -> "PartiallyCoherentWavefront":
        """Move modes, weights, and sim_params to a new device or dtype."""
        return PartiallyCoherentWavefront(
            Wavefront(self.modes.to(*args, **kwargs)),
            self.weights.to(*args, **kwargs),
            self.sim_params.to(*args, **kwargs),
            self.sim_params_modes.to(*args, **kwargs),
        )

    def __repr__(self) -> str:
        M = len(self.weights)
        H, W = self.sim_params.axis_sizes(("y", "x"))
        return (
            f"PartiallyCoherentWavefront("
            f"M={M}, grid=({H}, {W}), "
            f"total_power={self.total_power.item():.4g}, "
            f"device={self.modes.device})"
        )
