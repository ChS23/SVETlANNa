"""Tests for gamma.py: SpatialCoherence and coherence_forward."""
import pytest
import torch

from svetlanna import SimulationParameters
from svetlanna.wavefront import Wavefront
from svetlanna.coherence.gamma import SpatialCoherence, coherence_forward, _adjoint
from svetlanna.coherence import PartiallyCoherentWavefront
from svetlanna.coherence.wavefront_pc import _make_sim_params_modes


# ── helpers ───────────────────────────────────────────────────────────────────

def make_sim_params(H=8, W=8):
    return SimulationParameters(
        x=torch.linspace(-1e-3, 1e-3, W),
        y=torch.linspace(-1e-3, 1e-3, H),
        wavelength=torch.tensor(0.5e-3),
    )


def rank1_gamma(H=8, W=8):
    """Return (sp, Γ) for a rank-1 coherence function from a random mode."""
    sp = make_sim_params(H, W)
    phi = torch.randn(H * W, dtype=torch.complex64)
    phi = phi / phi.norm()
    lam = 3.0
    gamma_flat = lam * phi.conj().unsqueeze(1) * phi.unsqueeze(0)   # (HW, HW)
    return sp, gamma_flat.reshape(H, W, H, W)


# ── _adjoint ──────────────────────────────────────────────────────────────────

class TestAdjoint:
    def test_adjoint_involution(self):
        """(Γ†)† = Γ."""
        g = torch.randn(4, 4, 4, 4, dtype=torch.complex64)
        torch.testing.assert_close(_adjoint(_adjoint(g)), g)

    def test_adjoint_formula(self):
        """[Γ†](r1,r2) = Γ*(r2,r1): check a specific element."""
        g = torch.randn(3, 4, 3, 4, dtype=torch.complex64)
        ga = _adjoint(g)
        # ga[y1,x1,y2,x2] == g[y2,x2,y1,x1].conj()
        torch.testing.assert_close(ga[1, 2, 0, 3], g[0, 3, 1, 2].conj())

    def test_adjoint_hermitian_fixed_point(self):
        """A Hermitian Γ satisfies Γ† = Γ."""
        sp, gamma = rank1_gamma()
        torch.testing.assert_close(_adjoint(gamma), gamma, atol=1e-6, rtol=1e-6)


# ── SpatialCoherence ──────────────────────────────────────────────────────────

class TestSpatialCoherence:
    def test_construction(self):
        sp, gamma = rank1_gamma()
        sc = SpatialCoherence(gamma, sp)
        assert sc is not None

    def test_size_check_enforced(self):
        sp = make_sim_params(H=65, W=65)
        gamma = torch.zeros(65, 65, 65, 65, dtype=torch.complex64)
        with pytest.raises(ValueError, match="exceeds the 64"):
            SpatialCoherence(gamma, sp)

    def test_size_check_override(self):
        sp = make_sim_params(H=65, W=65)
        gamma = torch.zeros(65, 65, 65, 65, dtype=torch.complex64)
        sc = SpatialCoherence(gamma, sp, check_size=False)
        assert sc is not None

    def test_shape_mismatch_raises(self):
        sp = make_sim_params(H=8, W=8)
        gamma = torch.zeros(8, 8, 4, 4, dtype=torch.complex64)
        with pytest.raises(ValueError, match="does not match"):
            SpatialCoherence(gamma, sp)

    def test_intensity_shape(self):
        sp, gamma = rank1_gamma(H=6, W=7)
        sc = SpatialCoherence(gamma, sp)
        assert sc.intensity.shape == (6, 7)

    def test_intensity_non_negative(self):
        sp, gamma = rank1_gamma()
        sc = SpatialCoherence(gamma, sp)
        assert (sc.intensity >= 0).all()

    def test_intensity_is_diagonal(self):
        """Intensity = diagonal of Γ, which for rank-1 = λ |φ(r)|²."""
        sp, gamma = rank1_gamma(H=6, W=6)
        sc = SpatialCoherence(gamma, sp)
        # gamma = 3 * phi* ⊗ phi  → diagonal = 3 * |phi|²
        H, W = 6, 6
        g_mat = gamma.reshape(H * W, H * W)
        diag_expected = g_mat.diagonal().real.reshape(H, W)
        torch.testing.assert_close(sc.intensity, diag_expected)

    def test_repr(self):
        sp, gamma = rank1_gamma()
        sc = SpatialCoherence(gamma, sp)
        r = repr(sc)
        assert "SpatialCoherence" in r

    def test_from_pcw(self):
        sp = make_sim_params(H=8, W=8)
        M = 3
        spm = _make_sim_params_modes(sp, M)
        phi = torch.randn(M, 8, 8, dtype=torch.complex64)
        w = torch.rand(M) + 0.1
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)
        sc = SpatialCoherence.from_pcw(pcw)
        # Intensity should match pcw.intensity
        torch.testing.assert_close(sc.intensity, pcw.intensity, atol=1e-5, rtol=1e-5)


# ── coherence_forward ─────────────────────────────────────────────────────────

class TestCoherenceForward:
    def _make_phase_element(self, H, W, phase_value):
        """Element that multiplies by exp(i * phase_value) uniformly."""
        import torch.nn as nn

        class UniformPhase(nn.Module):
            def __init__(self, phi):
                super().__init__()
                self._phi = phi

            def forward(self, x):
                return Wavefront(torch.Tensor(x) * torch.exp(1j * torch.tensor(self._phi)))

        return UniformPhase(phase_value)

    def test_phase_element_analytic(self):
        """U = exp(i*theta)*I: coherence_forward should return unchanged Γ (|U|=1)."""
        sp, gamma = rank1_gamma(H=4, W=4)
        sc = SpatialCoherence(gamma, sp)
        elem = self._make_phase_element(4, 4, 0.7)
        sc_out = coherence_forward(elem, sc)
        # For a global phase, Γ' = e^{iθ} Γ e^{-iθ} = Γ
        torch.testing.assert_close(sc_out.data.abs(), sc.data.abs(), atol=1e-5, rtol=1e-5)

    def test_output_is_spatial_coherence(self):
        sp, gamma = rank1_gamma(H=4, W=4)
        sc = SpatialCoherence(gamma, sp)
        elem = self._make_phase_element(4, 4, 1.0)
        sc_out = coherence_forward(elem, sc)
        assert isinstance(sc_out, SpatialCoherence)

    def test_output_intensity_non_negative(self):
        sp, gamma = rank1_gamma(H=4, W=4)
        sc = SpatialCoherence(gamma, sp)
        elem = self._make_phase_element(4, 4, 1.2)
        sc_out = coherence_forward(elem, sc)
        assert (sc_out.intensity >= -1e-6).all()

    def test_diagonal_is_real(self):
        """Diagonal of Γ (intensity) must be real after propagation."""
        sp, gamma = rank1_gamma(H=4, W=4)
        sc = SpatialCoherence(gamma, sp)
        elem = self._make_phase_element(4, 4, 0.3)
        sc_out = coherence_forward(elem, sc)
        H, W = 4, 4
        g_mat = sc_out.data.reshape(H * W, H * W)
        diag = g_mat.diagonal()
        assert diag.imag.abs().max() < 1e-5

    def test_amplitude_element_reduces_power(self):
        """An attenuating aperture (amplitude < 1) reduces total coherence trace."""
        sp, gamma = rank1_gamma(H=4, W=4)
        sc = SpatialCoherence(gamma, sp)
        import torch.nn as nn

        class HalfAmplitude(nn.Module):
            def forward(self, x):
                return Wavefront(torch.Tensor(x) * 0.5)

        elem = HalfAmplitude()
        sc_out = coherence_forward(elem, sc)
        H, W = 4, 4
        trace_before = sc.data.reshape(H * W, H * W).diagonal().real.sum()
        trace_after = sc_out.data.reshape(H * W, H * W).diagonal().real.sum()
        # 0.5 * 0.5 = 0.25 factor on Γ
        torch.testing.assert_close(trace_after, 0.25 * trace_before, rtol=1e-4, atol=1e-8)

    def test_pcw_and_gamma_agree(self):
        """coherence_forward(phase) applied to rank-1 Γ agrees with PCW.through()."""
        import torch.nn as nn

        sp = make_sim_params(H=6, W=6)
        M = 2
        spm = _make_sim_params_modes(sp, M)
        torch.manual_seed(0)
        phi = torch.randn(M, 6, 6, dtype=torch.complex64)
        w = torch.rand(M).abs() + 0.1
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)

        class SpatialPhase(nn.Module):
            def __init__(self):
                super().__init__()
                self.phase = torch.randn(6, 6)

            def forward(self, x):
                return Wavefront(torch.Tensor(x) * torch.exp(1j * self.phase))

        elem = SpatialPhase()

        # PCW route
        pcw_out = pcw.through(elem)
        I_pcw = pcw_out.intensity   # (6, 6)

        # Γ route
        gamma_in = pcw.coherence_matrix()
        sc = SpatialCoherence(gamma_in, sp)
        sc_out = coherence_forward(elem, sc)
        I_gamma = sc_out.intensity   # (6, 6)

        torch.testing.assert_close(I_gamma, I_pcw, atol=1e-4, rtol=1e-4)
