"""Tests for sampling.py: stochastic field realizations."""
import pytest
import torch

from svetlanna import SimulationParameters
from svetlanna.wavefront import Wavefront
from svetlanna.coherence import PartiallyCoherentWavefront, sample_realizations, incoherent_source, detect_realizations
from svetlanna.coherence.wavefront_pc import _make_sim_params_modes


# ── helpers ───────────────────────────────────────────────────────────────────

def make_sim_params(H=16, W=16):
    return SimulationParameters(
        x=torch.linspace(-1e-3, 1e-3, W),
        y=torch.linspace(-1e-3, 1e-3, H),
        wavelength=torch.tensor(0.5e-3),
    )


def make_pcw(M=4, H=16, W=16, seed=42):
    sp = make_sim_params(H, W)
    spm = _make_sim_params_modes(sp, M)
    g = torch.Generator()
    g.manual_seed(seed)
    phi = torch.randn(M, H, W, generator=g) + 1j * torch.randn(M, H, W, generator=g)
    w = torch.rand(M, generator=g).abs() + 0.1
    return PartiallyCoherentWavefront(Wavefront(phi), w, sp, spm)


# ── sample_realizations ───────────────────────────────────────────────────────

class TestSampleRealizations:
    def test_shape(self):
        pcw = make_pcw(M=3, H=8, W=10)
        realizations = sample_realizations(pcw, k=5)
        assert realizations.shape == (5, 8, 10)

    def test_complex_output(self):
        pcw = make_pcw()
        realizations = sample_realizations(pcw, k=4)
        assert realizations.is_complex()

    def test_k_zero_raises(self):
        pcw = make_pcw()
        with pytest.raises(ValueError, match="k must be >= 1"):
            sample_realizations(pcw, k=0)

    def test_generator_reproducibility(self):
        pcw = make_pcw()
        gen1 = torch.Generator()
        gen1.manual_seed(99)
        r1 = sample_realizations(pcw, k=10, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(99)
        r2 = sample_realizations(pcw, k=10, generator=gen2)

        torch.testing.assert_close(r1, r2)

    def test_different_seeds_differ(self):
        pcw = make_pcw()
        gen1 = torch.Generator().manual_seed(1)
        gen2 = torch.Generator().manual_seed(2)
        r1 = sample_realizations(pcw, k=4, generator=gen1)
        r2 = sample_realizations(pcw, k=4, generator=gen2)
        assert not torch.allclose(torch.Tensor(r1), torch.Tensor(r2))

    def test_ensemble_mean_converges_to_intensity(self):
        """Ensemble average |ψ^(k)|² → pcw.intensity with large k.

        Uses 2000 realizations; O(k^{-1/2}) error ≈ 2%, so rtol=0.15 is generous.
        """
        pcw = make_pcw(M=4, H=12, W=12, seed=0)
        gen = torch.Generator().manual_seed(7)
        k = 2000
        realizations = sample_realizations(pcw, k=k, generator=gen)
        I_mc = torch.Tensor(realizations).abs().pow(2).mean(dim=0)
        I_exact = pcw.intensity
        torch.testing.assert_close(I_mc, I_exact, atol=0.0, rtol=0.15)

    def test_pcw_method_matches_module_function(self):
        pcw = make_pcw()
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        r1 = pcw.sample_realizations(k=5, generator=gen1)
        r2 = sample_realizations(pcw, k=5, generator=gen2)
        torch.testing.assert_close(torch.Tensor(r1), torch.Tensor(r2))


# ── incoherent_source ─────────────────────────────────────────────────────────

class TestIncoherentSource:
    def test_shape(self):
        sp = make_sim_params(H=8, W=10)
        I = torch.rand(8, 10)
        realizations = incoherent_source(sp, I, k=5)
        assert realizations.shape == (5, 8, 10)

    def test_complex_output(self):
        sp = make_sim_params()
        I = torch.ones(16, 16)
        r = incoherent_source(sp, I, k=3)
        assert r.is_complex()

    def test_amplitude_matches_sqrt_intensity(self):
        """Each realization should have |ψ(r)| = sqrt(I(r))."""
        sp = make_sim_params(H=6, W=6)
        I = torch.rand(6, 6) + 0.1
        r = incoherent_source(sp, I, k=10)
        amp = torch.Tensor(r).abs()   # (10, 6, 6)
        sqrt_I = I.sqrt().unsqueeze(0).expand_as(amp)
        torch.testing.assert_close(amp, sqrt_I, atol=1e-5, rtol=1e-5)

    def test_k_zero_raises(self):
        sp = make_sim_params()
        I = torch.ones(16, 16)
        with pytest.raises(ValueError, match="k must be >= 1"):
            incoherent_source(sp, I, k=0)

    def test_generator_reproducibility(self):
        sp = make_sim_params()
        I = torch.rand(16, 16)
        gen1 = torch.Generator().manual_seed(3)
        gen2 = torch.Generator().manual_seed(3)
        r1 = incoherent_source(sp, I, k=5, generator=gen1)
        r2 = incoherent_source(sp, I, k=5, generator=gen2)
        torch.testing.assert_close(torch.Tensor(r1), torch.Tensor(r2))

    def test_negative_intensity_clamped(self):
        """Negative-intensity pixels are treated as zero (no complex-amp issue)."""
        sp = make_sim_params(H=4, W=4)
        I = torch.full((4, 4), -1.0)
        r = incoherent_source(sp, I, k=2)
        assert (torch.Tensor(r).abs() == 0).all()


# ── detect_realizations ───────────────────────────────────────────────────────

class TestDetectRealizations:
    def _identity_net(self):
        import torch.nn as nn
        class Identity(nn.Module):
            def forward(self, x):
                return x
        return Identity()

    def test_shape(self):
        sp = make_sim_params(H=8, W=8)
        I = torch.rand(8, 8)
        r = incoherent_source(sp, I, k=10)
        I_det = detect_realizations(r, self._identity_net())
        assert I_det.shape == (8, 8)

    def test_identity_network_mean_matches_intensity(self):
        """For an identity network, mean(|ψ|²) ≈ original intensity (large k)."""
        sp = make_sim_params(H=8, W=8)
        I = torch.rand(8, 8) + 0.1
        gen = torch.Generator().manual_seed(0)
        k = 2000
        r = incoherent_source(sp, I, k=k, generator=gen)
        I_det = detect_realizations(r, self._identity_net())
        torch.testing.assert_close(I_det, I, atol=0.0, rtol=0.15)

    def test_custom_weights(self):
        """Custom uniform weights == unweighted mean."""
        sp = make_sim_params(H=4, W=4)
        I = torch.ones(4, 4)
        r = incoherent_source(sp, I, k=5)
        w = torch.ones(5)
        I_weighted = detect_realizations(r, self._identity_net(), weights=w)
        I_unweighted = detect_realizations(r, self._identity_net())
        torch.testing.assert_close(I_weighted, I_unweighted)

    def test_weights_normalize(self):
        """Weights are normalized internally; doubling them has no effect."""
        sp = make_sim_params(H=4, W=4)
        I = torch.ones(4, 4)
        r = incoherent_source(sp, I, k=5)
        w = torch.ones(5)
        I1 = detect_realizations(r, self._identity_net(), weights=w)
        I2 = detect_realizations(r, self._identity_net(), weights=2 * w)
        torch.testing.assert_close(I1, I2)
