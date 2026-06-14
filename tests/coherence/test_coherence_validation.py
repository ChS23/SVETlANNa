"""Physics validation tests for partial coherence.

Tests verify that the three representations (mode decomposition, full Γ tensor,
Monte Carlo sampling) are mutually consistent and match analytic predictions.
"""
import math
import pytest
import torch

from svetlanna import SimulationParameters
from svetlanna.wavefront import Wavefront
from svetlanna.coherence import (
    PartiallyCoherentWavefront,
    gaussian_schell_model,
    modes_from_coherence_matrix,
    SpatialCoherence,
    coherence_forward,
    sample_realizations,
)
from svetlanna.coherence.wavefront_pc import _make_sim_params_modes


# ── helpers ───────────────────────────────────────────────────────────────────

def make_sim_params(H=32, W=32, lam=0.5e-3, size=4e-3):
    return SimulationParameters(
        x=torch.linspace(-size / 2, size / 2, W),
        y=torch.linspace(-size / 2, size / 2, H),
        wavelength=torch.tensor(lam),
    )


class SpatialPhaseElement(torch.nn.Module):
    """Thin element with a fixed spatial phase mask (not trainable)."""
    def __init__(self, phase: torch.Tensor):
        super().__init__()
        self.register_buffer("_phase", phase)

    def forward(self, x: Wavefront) -> Wavefront:
        return Wavefront(torch.Tensor(x) * torch.exp(1j * self._phase))


# ── TEST 1: Coherent limit ────────────────────────────────────────────────────

class TestCoherentLimit:
    def test_single_mode_intensity_matches_wavefront(self):
        """PCW with M=1 mode should give same intensity as the coherent Wavefront."""
        sp = make_sim_params(H=16, W=16)
        # Build a random coherent wavefront
        phi = torch.randn(16, 16) + 1j * torch.randn(16, 16)
        wf = Wavefront(phi.to(torch.complex64))

        # Wrap it as a PCW with one mode, weight=1
        spm = _make_sim_params_modes(sp, 1)
        pcw = PartiallyCoherentWavefront(
            Wavefront(phi.unsqueeze(0).to(torch.complex64)),
            torch.tensor([1.0]),
            sp,
            spm,
        )
        # Intensities should match
        torch.testing.assert_close(pcw.intensity, wf.intensity, atol=1e-6, rtol=1e-6)

    def test_gsm_coherent_limit(self):
        """GSM with very large coherence_width → M=1 mode."""
        sp = make_sim_params()
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=1e6,
                                     power_fraction=0.99)
        assert len(pcw.weights) == 1

    def test_through_coherent_limit_identity(self):
        """Propagating a M=1 PCW through an identity does not change intensity."""
        sp = make_sim_params(H=8, W=8)
        spm = _make_sim_params_modes(sp, 1)
        phi = (torch.randn(1, 8, 8) + 1j * torch.randn(1, 8, 8)).to(torch.complex64)
        pcw = PartiallyCoherentWavefront(Wavefront(phi), torch.tensor([2.0]), sp, spm)
        phase = torch.zeros(8, 8)
        out = pcw.through(SpatialPhaseElement(phase))
        torch.testing.assert_close(out.intensity, pcw.intensity, atol=1e-5, rtol=1e-5)


# ── TEST 2: Three-representation consistency ──────────────────────────────────

class TestThreeRepresentations:
    """Mode decomposition, full Γ, and Monte Carlo must give the same I_det."""

    @pytest.fixture
    def pcw_and_element(self):
        sp = make_sim_params(H=12, W=12)
        pcw = gaussian_schell_model(
            sp, waist_radius=1.5e-3, coherence_width=0.8e-3, power_fraction=0.99
        )
        torch.manual_seed(0)
        phase = torch.randn(12, 12) * 0.5
        elem = SpatialPhaseElement(phase)
        return pcw, elem, sp

    def test_mode_vs_gamma_intensity(self, pcw_and_element):
        """PCW.through() intensity agrees with coherence_forward() diagonal."""
        pcw, elem, sp = pcw_and_element
        # Mode route
        I_pcw = pcw.through(elem).intensity    # (H, W)
        # Γ route
        gamma_in = pcw.coherence_matrix()
        sc = SpatialCoherence(gamma_in, sp)
        sc_out = coherence_forward(elem, sc)
        I_gamma = sc_out.intensity             # (H, W)

        torch.testing.assert_close(I_gamma, I_pcw, atol=1e-4, rtol=1e-3)

    def test_mode_vs_montecarlo_intensity(self, pcw_and_element):
        """PCW.through() intensity agrees with Monte Carlo ensemble mean (rtol≤0.15)."""
        pcw, elem, sp = pcw_and_element
        # Mode route
        I_exact = pcw.through(elem).intensity   # (H, W)

        # Monte Carlo route
        gen = torch.Generator().manual_seed(42)
        k = 3000
        realizations = sample_realizations(pcw, k=k, generator=gen)  # (k, H, W)
        out_mc = torch.Tensor(elem(realizations))
        I_mc = out_mc.abs().pow(2).mean(dim=0)   # (H, W)

        torch.testing.assert_close(I_mc, I_exact, atol=0.0, rtol=0.15)


# ── TEST 3: Power conservation through unitary elements ───────────────────────

class TestPowerConservation:
    def test_phase_element_preserves_total_power(self):
        """A pure phase element (|U|=1 everywhere) must not change total power."""
        sp = make_sim_params(H=16, W=16)
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)
        phase = torch.randn(16, 16)
        out = pcw.through(SpatialPhaseElement(phase))

        dx = ((sp.x[-1] - sp.x[0]).abs() / (len(sp.x) - 1)).item()
        dy = ((sp.y[-1] - sp.y[0]).abs() / (len(sp.y) - 1)).item()
        power_in = (pcw.intensity * dx * dy).sum()
        power_out = (out.intensity * dx * dy).sum()
        # Phase element preserves power: relative error < 0.01%
        torch.testing.assert_close(power_out, power_in, rtol=1e-4, atol=0.0)

    def test_attenuating_aperture_reduces_power(self):
        """Half-amplitude aperture should give ~0.25× power."""
        sp = make_sim_params(H=16, W=16)
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)

        class HalfAmplitude(torch.nn.Module):
            def forward(self, x):
                return Wavefront(torch.Tensor(x) * 0.5)

        out = pcw.through(HalfAmplitude())
        # Γ' = 0.25 Γ so I' = 0.25 I
        torch.testing.assert_close(out.intensity, 0.25 * pcw.intensity, atol=1e-6, rtol=1e-5)


# ── TEST 4: GSM diagnostic properties ────────────────────────────────────────

class TestGSMProperties:
    def test_intensity_is_gaussian(self):
        """GSM source intensity profile should be Gaussian-shaped (peak at center)."""
        sp = make_sim_params(H=32, W=32, size=6e-3)
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3,
                                     power_fraction=0.999)
        I = pcw.intensity
        Hmid, Wmid = I.shape[0] // 2, I.shape[1] // 2
        # Center pixel should be the maximum
        assert I[Hmid, Wmid] >= I.max() * 0.9

    def test_coherence_width_affects_mode_count(self):
        """Smaller coherence_width → more modes needed."""
        sp = make_sim_params()
        pcw_wide = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=2e-3,
                                          power_fraction=0.99)
        pcw_narrow = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.2e-3,
                                            power_fraction=0.99)
        assert len(pcw_narrow.weights) > len(pcw_wide.weights)

    def test_gsm_degree_of_coherence_at_origin(self):
        """μ(r, r) = 1: the degree of coherence is always 1 for coincident points."""
        sp = make_sim_params(H=16, W=16)
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)
        mu = pcw.degree_of_coherence((8, 8), (8, 8))
        assert abs(mu.abs().item() - 1.0) < 1e-4

    def test_degree_of_coherence_magnitude_bounded(self):
        """|μ(r1, r2)| ≤ 1 for all point pairs."""
        sp = make_sim_params(H=16, W=16)
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.5e-3)
        # Sample a few point pairs
        for (y1, x1), (y2, x2) in [((4, 4), (12, 12)), ((8, 4), (8, 12)), ((2, 2), (14, 14))]:
            mu = pcw.degree_of_coherence((y1, x1), (y2, x2))
            assert mu.abs().item() <= 1.0 + 1e-5


# ── TEST 5: Young's double-slit visibility ────────────────────────────────────

class TestYoungVisibility:
    def test_visibility_decreases_with_slit_separation(self):
        """For a GSM source, visibility V = |μ(d)| decreases with slit separation."""
        sp = make_sim_params(H=32, W=32, size=4e-3)
        pcw = gaussian_schell_model(sp, waist_radius=1e-3, coherence_width=0.8e-3,
                                     power_fraction=0.999)
        H, W = 32, 32
        center = W // 2

        # Small separation (near center) → high coherence
        mu_small = pcw.degree_of_coherence((H // 2, center - 2), (H // 2, center + 2))
        # Large separation → lower coherence
        mu_large = pcw.degree_of_coherence((H // 2, center - 8), (H // 2, center + 8))

        assert mu_small.abs().item() > mu_large.abs().item()

    def test_visibility_gsm_analytic_trend(self):
        """|μ_GSM(Δr)| = exp(-Δr²/(2σc²)): compare to analytic at two separations."""
        sigma_c = 0.8e-3
        sp = make_sim_params(H=32, W=32, size=4e-3)
        pcw = gaussian_schell_model(sp, waist_radius=2e-3, coherence_width=sigma_c,
                                     power_fraction=0.999)
        H, W = 32, 32
        dx = (sp.x[-1] - sp.x[0]).abs().item() / (W - 1)
        center = W // 2

        for sep_pix in [2, 5]:
            sep_m = sep_pix * dx * 2   # separation in meters (2 points apart)
            mu_numeric = pcw.degree_of_coherence(
                (H // 2, center - sep_pix), (H // 2, center + sep_pix)
            )
            mu_analytic = math.exp(-sep_m ** 2 / (2 * sigma_c ** 2))
            # Allow 20% tolerance (limited mode truncation, grid effects)
            assert abs(mu_numeric.abs().item() - mu_analytic) < 0.20
