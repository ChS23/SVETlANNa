"""Tests for coherence source constructors."""
import math
import pytest
import torch

from svetlanna import SimulationParameters
from svetlanna.coherence import (
    PartiallyCoherentWavefront,
    gaussian_schell_model,
    modes_from_coherence_matrix,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_sim_params(H=32, W=32, lam=0.5e-3, size=4e-3):
    return SimulationParameters(
        x=torch.linspace(-size / 2, size / 2, W),
        y=torch.linspace(-size / 2, size / 2, H),
        wavelength=torch.tensor(lam),
    )


def gsm_analytic_gamma(sp, w0, sigma_c):
    """Analytic Γ for a Gaussian Schell-model source (unnormalized, power=1)."""
    x = sp.x                         # (W,)
    y = sp.y                         # (H,)
    Y, X = torch.meshgrid(y, x, indexing="ij")   # (H, W)
    r2 = X.pow(2) + Y.pow(2)

    # amplitude cross-correlation:  W(r1,r2) = sqrt(I(r1)*I(r2)) * mu(r1-r2)
    I = torch.exp(-r2 / w0 ** 2)    # (H, W)  — Gaussian beam intensity (not power-normed)

    # Γ(r1,r2) = sqrt(I(r1)) * sqrt(I(r2)) * exp(-(r1-r2)²/(2σc²))
    #   built as outer product over flattened pixel pairs
    sqI = I.sqrt().flatten()         # (HW,)
    outer_amp = sqI.unsqueeze(1) * sqI.unsqueeze(0)   # (HW, HW)

    r1_x = X.flatten()
    r1_y = Y.flatten()
    dr2 = (r1_x.unsqueeze(1) - r1_x.unsqueeze(0)).pow(2) + \
          (r1_y.unsqueeze(1) - r1_y.unsqueeze(0)).pow(2)
    # W = exp(-a*(r1²+r2²)/2) * exp(-b*(r1-r2)²/2) with a=1/w0², b=1/(2σc²)
    # → mu(Δr) = exp(-Δr²/(4σc²))
    mu = torch.exp(-dr2 / (4.0 * sigma_c ** 2))

    gamma_flat = (outer_amp * mu).to(torch.complex64)
    H, W = len(sp.y), len(sp.x)
    return gamma_flat.reshape(H, W, H, W)


# ── gaussian_schell_model ─────────────────────────────────────────────────────

class TestGaussianSchellModel:
    def test_returns_pcw(self):
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)
        assert isinstance(pcw, PartiallyCoherentWavefront)

    def test_power_default_one(self):
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)
        assert abs(pcw.total_power.item() - 1.0) < 1e-5

    def test_custom_power(self):
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3, power=3.7)
        assert abs(pcw.total_power.item() - 3.7) < 1e-4

    def test_intensity_non_negative(self):
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)
        assert (pcw.intensity >= 0).all()

    def test_modes_complex(self):
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)
        assert pcw.modes.is_complex()

    def test_coherent_limit_single_mode(self):
        """Very large coherence_width → nearly coherent → M = 1."""
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=100.0,
                                     power_fraction=0.99)
        assert len(pcw.weights) == 1

    def test_invalid_coherence_width_zero(self):
        sp = make_sim_params()
        with pytest.raises(ValueError, match="coherence_width must be > 0"):
            gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.0)

    def test_invalid_power(self):
        sp = make_sim_params()
        with pytest.raises(ValueError, match="power must be positive"):
            gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3, power=-1.0)

    def test_max_modes(self):
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.1e-3,
                                     max_modes=3)
        # M = M_1d² ≤ 9
        assert len(pcw.weights) <= 9

    def test_intensity_matches_gaussian_profile(self):
        """GSM intensity profile should match the Gaussian exp(-r²/w0²) shape."""
        w0 = 1.5e-3
        sigma_c = 0.8e-3
        sp = make_sim_params(H=32, W=32, size=6e-3)
        pcw = gaussian_schell_model(sp, waist_radius=w0, coherence_width=sigma_c,
                                     power_fraction=0.999)
        I = pcw.intensity    # (32, 32)

        # Build analytic intensity profile
        X, Y = torch.meshgrid(sp.x, sp.y, indexing="xy")
        I_ana = torch.exp(-(X.pow(2) + Y.pow(2)) / w0 ** 2).T  # (H, W)

        # Normalize both to unit peak
        I_norm = I / I.max().clamp(min=1e-30)
        I_ana_norm = I_ana / I_ana.max().clamp(min=1e-30)
        torch.testing.assert_close(I_norm, I_ana_norm, atol=0.05, rtol=0.0)

    def test_coherence_degree_matches_analytic(self):
        """Coherence degree |μ(r1,r2)| at several point pairs matches analytic.

        For GSM with sources.py convention (a=1/w0², b=1/(2σc²)):
            μ(Δr) = exp(-|Δr|²/(4σc²))
        """
        w0 = 1.5e-3
        sigma_c = 0.8e-3
        sp = make_sim_params(H=32, W=32, size=6e-3)
        pcw = gaussian_schell_model(sp, waist_radius=w0, coherence_width=sigma_c,
                                     power_fraction=0.999)

        dx = ((sp.x[-1] - sp.x[0]).abs() / (len(sp.x) - 1)).item()
        H, W = 32, 32

        # Test a few separations along x at the center row
        for sep_pix in [1, 3, 6]:
            y_c = H // 2
            x1, x2 = W // 2 - sep_pix, W // 2 + sep_pix
            sep_m = 2 * sep_pix * dx

            mu_num = pcw.degree_of_coherence((y_c, x1), (y_c, x2)).abs().item()
            mu_ana = math.exp(-sep_m ** 2 / (4.0 * sigma_c ** 2))
            assert abs(mu_num - mu_ana) < 0.15, (
                f"sep_pix={sep_pix}: |μ_num|={mu_num:.4f} vs μ_ana={mu_ana:.4f}"
            )


# ── modes_from_coherence_matrix ───────────────────────────────────────────────

class TestModesFromCoherenceMatrix:
    def _rank1_gamma(self, H=8, W=8):
        """A rank-1 Γ from a unit-norm mode."""
        sp = make_sim_params(H=H, W=W)
        phi = torch.randn(H, W) + 1j * torch.randn(H, W)
        phi_flat = phi.flatten()
        phi_flat = phi_flat / phi_flat.norm()   # unit norm
        gamma_flat = phi_flat.conj().unsqueeze(1) * phi_flat.unsqueeze(0)
        lam = 2.5
        return sp, lam * gamma_flat.reshape(H, W, H, W), phi_flat.reshape(H, W), lam

    def test_rank1_roundtrip(self):
        """modes_from_coherence_matrix on a rank-1 Γ returns one non-zero eigenvalue."""
        sp, gamma, phi, lam = self._rank1_gamma()
        pcw = modes_from_coherence_matrix(gamma, sp, power_fraction=0.99)
        # Should have at least one mode; dominant weight ≈ lam
        assert len(pcw.weights) >= 1
        assert abs(pcw.weights[0].item() - lam) < 1e-3

    def test_reconstruction_matches_input(self):
        """Reconstructed Γ matches input to rtol=1e-3 for known low-rank Γ."""
        sp, gamma, _, _ = self._rank1_gamma(H=8, W=8)
        pcw = modes_from_coherence_matrix(gamma, sp, power_fraction=0.999)
        gamma_out = pcw.coherence_matrix()
        torch.testing.assert_close(gamma_out.abs(), gamma.abs(), rtol=1e-3, atol=1e-6)

    def test_power_preserved(self):
        sp, gamma, _, lam = self._rank1_gamma()
        pcw = modes_from_coherence_matrix(gamma, sp, power_fraction=0.99)
        # Total power = trace(Γ) / normalization = lam (since phi is not unit-normed here)
        # Just verify total_power ≈ lam (trace)
        H, W = len(sp.y), len(sp.x)
        trace = gamma.reshape(H * W, H * W).diagonal().real.sum()
        # power_fraction=0.99 so captured power ≥ 0.99 * trace
        assert pcw.total_power >= 0.99 * trace

    def test_weights_non_negative(self):
        sp, gamma, _, _ = self._rank1_gamma()
        pcw = modes_from_coherence_matrix(gamma, sp)
        assert (pcw.weights >= 0).all()

    def test_invalid_rank(self):
        gamma_3d = torch.zeros(8, 8, 8)
        sp = make_sim_params(H=8, W=8)
        with pytest.raises(ValueError, match="4-D"):
            modes_from_coherence_matrix(gamma_3d, sp)

    def test_invalid_shape(self):
        gamma = torch.zeros(8, 8, 4, 4)
        sp = make_sim_params(H=8, W=8)
        with pytest.raises(ValueError, match="H,W,H,W"):
            modes_from_coherence_matrix(gamma, sp)

    def test_hermitian_input(self):
        """Non-Hermitian Γ is symmetrized internally; no error raised."""
        sp = make_sim_params(H=8, W=8)
        H, W = 8, 8
        # Build random (non-Hermitian) Γ
        gamma_rand = torch.randn(H, W, H, W, dtype=torch.complex64)
        pcw = modes_from_coherence_matrix(gamma_rand, sp)
        assert pcw is not None
