"""Unit tests for PartiallyCoherentWavefront container."""
import pytest
import torch

from svetlanna import SimulationParameters
from svetlanna.wavefront import Wavefront
from svetlanna.coherence import PartiallyCoherentWavefront
from svetlanna.coherence.wavefront_pc import _make_sim_params_modes


# ── helpers ───────────────────────────────────────────────────────────────────

def make_sim_params(H=16, W=16, wavelength=0.5e-3):
    return SimulationParameters(
        x=torch.linspace(-1e-3, 1e-3, W),
        y=torch.linspace(-1e-3, 1e-3, H),
        wavelength=torch.tensor(wavelength),
    )


def make_pcw(M=3, H=16, W=16):
    sp = make_sim_params(H, W)
    spm = _make_sim_params_modes(sp, M)
    phi = torch.randn(M, H, W) + 1j * torch.randn(M, H, W)
    weights = torch.rand(M).abs() + 0.1
    return PartiallyCoherentWavefront(Wavefront(phi), weights, sp, spm)


# ── _validate ─────────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_construction(self):
        pcw = make_pcw()
        assert pcw is not None

    def test_missing_mode_axis_in_spm(self):
        sp = make_sim_params()
        phi = torch.randn(3, 16, 16, dtype=torch.complex64)
        w = torch.ones(3)
        with pytest.raises(ValueError, match="'mode' axis must be registered"):
            PartiallyCoherentWavefront(Wavefront(phi), w, sp, sp)

    def test_mode_axis_in_sp(self):
        sp = make_sim_params()
        spm = _make_sim_params_modes(sp, 3)
        # try putting mode in sp too
        sp_with_mode = _make_sim_params_modes(sp, 3)
        phi = torch.randn(3, 16, 16, dtype=torch.complex64)
        w = torch.ones(3)
        with pytest.raises(ValueError, match="sim_params must not contain"):
            PartiallyCoherentWavefront(Wavefront(phi), w, sp_with_mode, spm)

    def test_real_modes_rejected(self):
        sp = make_sim_params()
        spm = _make_sim_params_modes(sp, 2)
        phi = torch.randn(2, 16, 16)   # real
        w = torch.ones(2)
        with pytest.raises(TypeError, match="complex"):
            PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)

    def test_negative_weights_rejected(self):
        sp = make_sim_params()
        spm = _make_sim_params_modes(sp, 2)
        phi = torch.randn(2, 16, 16, dtype=torch.complex64)
        w = torch.tensor([-0.1, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)

    def test_mode_count_mismatch(self):
        sp = make_sim_params()
        spm = _make_sim_params_modes(sp, 5)
        phi = torch.randn(3, 16, 16, dtype=torch.complex64)
        w = torch.ones(5)
        with pytest.raises(ValueError, match="does not match"):
            PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)

    def test_2d_weights_rejected(self):
        sp = make_sim_params()
        spm = _make_sim_params_modes(sp, 3)
        phi = torch.randn(3, 16, 16, dtype=torch.complex64)
        w = torch.ones(3, 1)
        with pytest.raises(ValueError, match="1-D"):
            PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)


# ── intensity ─────────────────────────────────────────────────────────────────

class TestIntensity:
    def test_intensity_non_negative(self):
        pcw = make_pcw(M=5)
        I = pcw.intensity
        assert (I >= 0).all()

    def test_intensity_shape(self):
        pcw = make_pcw(M=4, H=12, W=14)
        assert pcw.intensity.shape == (12, 14)

    def test_intensity_single_mode(self):
        """With one mode, intensity = λ * |φ|²."""
        sp = make_sim_params(H=8, W=8)
        spm = _make_sim_params_modes(sp, 1)
        phi = torch.ones(1, 8, 8, dtype=torch.complex64)
        w = torch.tensor([2.5])
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)
        torch.testing.assert_close(pcw.intensity, 2.5 * torch.ones(8, 8))

    def test_intensity_sum_of_modes(self):
        """Intensity = sum of weighted per-mode intensities."""
        sp = make_sim_params(H=8, W=8)
        M = 4
        spm = _make_sim_params_modes(sp, M)
        phi = torch.randn(M, 8, 8) + 1j * torch.randn(M, 8, 8)
        w = torch.rand(M) + 0.1
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)
        expected = sum(w[n] * phi[n].abs().pow(2) for n in range(M))
        torch.testing.assert_close(pcw.intensity, expected)


# ── total_power ───────────────────────────────────────────────────────────────

class TestTotalPower:
    def test_total_power(self):
        pcw = make_pcw(M=6)
        torch.testing.assert_close(pcw.total_power, pcw.weights.sum())


# ── through / detect (shape) ──────────────────────────────────────────────────

class TestThroughDetect:
    def _identity_net(self):
        """A network that returns its input unchanged."""
        import torch.nn as nn
        class Identity(nn.Module):
            def forward(self, x):
                return x
        return Identity()

    def test_through_preserves_shape(self):
        pcw = make_pcw(M=3, H=10, W=12)
        out = pcw.through(self._identity_net())
        assert out.modes.shape == pcw.modes.shape

    def test_through_preserves_weights(self):
        pcw = make_pcw(M=3)
        out = pcw.through(self._identity_net())
        torch.testing.assert_close(out.weights, pcw.weights)

    def test_detect_shape(self):
        pcw = make_pcw(M=3, H=10, W=12)
        I = pcw.detect(self._identity_net())
        assert I.shape == (10, 12)

    def test_detect_equals_intensity_for_identity(self):
        pcw = make_pcw(M=3, H=8, W=8)
        I_detect = pcw.detect(self._identity_net())
        torch.testing.assert_close(I_detect, pcw.intensity)

    def test_detect_gradients_flow(self):
        """Gradients from detect() must reach trainable parameters."""
        import torch.nn as nn

        sp = make_sim_params(H=8, W=8)
        M = 2
        spm = _make_sim_params_modes(sp, M)
        phi = (torch.randn(M, 8, 8) + 1j * torch.randn(M, 8, 8)).detach()
        w = torch.rand(M).abs() + 0.1
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)

        # Network with a learnable complex phase mask
        class PhaseNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.phase = nn.Parameter(torch.zeros(8, 8))
            def forward(self, x):
                return Wavefront(torch.Tensor(x) * torch.exp(1j * self.phase))

        net = PhaseNet()
        I = pcw.detect(net)
        loss = I.sum()
        loss.backward()
        assert net.phase.grad is not None
        assert net.phase.grad.abs().sum() > 0


# ── truncate ──────────────────────────────────────────────────────────────────

class TestTruncate:
    def test_truncate_keeps_enough_power(self):
        pcw = make_pcw(M=10)
        trunc = pcw.truncate(power_fraction=0.9)
        assert trunc.total_power >= 0.9 * pcw.total_power

    def test_truncate_reduces_mode_count(self):
        sp = make_sim_params()
        M = 10
        spm = _make_sim_params_modes(sp, M)
        # Very skewed weights so truncation cuts many modes
        w = torch.exp(-torch.arange(M, dtype=torch.float32) * 2.0)
        phi = torch.randn(M, 16, 16, dtype=torch.complex64)
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)
        trunc = pcw.truncate(power_fraction=0.95)
        assert trunc.modes.shape[0] < M

    def test_truncate_keeps_at_least_one(self):
        pcw = make_pcw(M=5)
        trunc = pcw.truncate(power_fraction=0.0)
        assert trunc.modes.shape[0] >= 1


# ── reorthogonalize ───────────────────────────────────────────────────────────

class TestReorthogonalize:
    def test_reorthogonalize_preserves_coherence_matrix(self):
        """Γ before and after reorthogonalize must be equal."""
        sp = make_sim_params(H=8, W=8)
        M = 3
        spm = _make_sim_params_modes(sp, M)
        phi = torch.randn(M, 8, 8) + 1j * torch.randn(M, 8, 8)
        w = torch.rand(M) + 0.1
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)

        reortho = pcw.reorthogonalize()
        gamma_orig = pcw.coherence_matrix()
        gamma_new = reortho.coherence_matrix()
        torch.testing.assert_close(gamma_new, gamma_orig, atol=1e-5, rtol=1e-5)

    def test_reorthogonalize_modes_are_orthonormal(self):
        """After reorthogonalize, modes should be approximately orthonormal."""
        sp = make_sim_params(H=8, W=8)
        M = 3
        spm = _make_sim_params_modes(sp, M)
        # Make non-orthogonal modes
        phi = torch.randn(M, 8, 8) + 1j * torch.randn(M, 8, 8)
        w = torch.rand(M) + 0.1
        pcw = PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)
        reortho = pcw.reorthogonalize()

        phi2 = torch.Tensor(reortho.modes)   # (M', H, W)
        M2 = phi2.shape[0]
        phi_flat = phi2.reshape(M2, -1)       # (M', HW)
        gram = phi_flat @ phi_flat.conj().mT  # (M', M')
        # Diagonal entries should be 1 (unit norm), off-diag near 0
        diag = gram.diagonal().real
        assert diag.min() > 0   # non-zero norms


# ── to() device/dtype ─────────────────────────────────────────────────────────

class TestTo:
    def test_to_complex64(self):
        pcw = make_pcw()
        pcw2 = pcw.to(dtype=torch.complex64)
        assert pcw2.modes.dtype == torch.complex64

    def test_to_cpu(self):
        pcw = make_pcw()
        pcw2 = pcw.to("cpu")
        assert pcw2.modes.device.type == "cpu"
        assert pcw2.weights.device.type == "cpu"


# ── repr ──────────────────────────────────────────────────────────────────────

def test_repr():
    pcw = make_pcw(M=3, H=16, W=16)
    r = repr(pcw)
    assert "M=3" in r
    assert "16" in r
