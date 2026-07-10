"""Tests for ensemble-axis reduction in Detector (reduce_axes) and
autograd correctness of DetectorProcessorClf.forward.

The convention under test: ensemble weights (mode eigenvalues, spectral
density, 1/K of Monte-Carlo) are folded into the field amplitudes as
sqrt(weight), so the detector reduction is a plain unweighted sum over
the named axes.
"""
import pytest
import torch
from torch import nn

from svetlanna import SimulationParameters
from svetlanna.wavefront import Wavefront
from svetlanna.detector import Detector, DetectorProcessorClf
from svetlanna.simulation_parameters import AxisNotFound


H, W = 8, 10


def make_sim_params(mode: int | None = None, wavelength=1e-6):
    """Axes declared (x, y, wavelength[, mode]) -> tensor layout (..., [M,] [Nwl,] H, W)."""
    axes: dict = {
        "x": torch.linspace(-1e-3, 1e-3, W),
        "y": torch.linspace(-1e-3, 1e-3, H),
        "wavelength": wavelength,
    }
    if mode is not None:
        axes["mode"] = torch.arange(mode)
    return SimulationParameters(axes)


def test_no_reduce_backward_compat():
    """Without reduce_axes the detector must keep every axis untouched."""
    detector = Detector(make_sim_params(), func="intensity")
    field = torch.rand(3, H, W) + 1j * torch.rand(3, H, W)
    out = detector.forward(Wavefront(field))
    assert out.shape == (3, H, W)
    assert torch.allclose(out, field.abs().pow(2))


def test_reduce_mode_axis_matches_weighted_sum():
    """Sum over folded sqrt(w)-modes == weighted sum of mode intensities."""
    M = 5
    sp = make_sim_params(mode=M)
    phi = torch.randn(M, H, W, dtype=torch.complex64)
    w = torch.rand(M)

    folded = Wavefront(w.sqrt()[:, None, None] * phi)
    detector = Detector(sp, func="intensity", reduce_axes=("mode",))
    out = detector.forward(folded)

    expected = (w[:, None, None] * phi.abs().pow(2)).sum(dim=0)
    assert out.shape == (H, W)
    assert torch.allclose(out, expected, atol=1e-6)


def test_reduce_keeps_leading_batch():
    """Batch axis (unregistered, leading) must survive the reduction."""
    B, M = 4, 3
    sp = make_sim_params(mode=M)
    field = torch.randn(B, M, H, W, dtype=torch.complex64)
    detector = Detector(sp, func="intensity", reduce_axes=("mode",))
    out = detector.forward(Wavefront(field))
    assert out.shape == (B, H, W)
    assert torch.allclose(out, field.abs().pow(2).sum(dim=1), atol=1e-6)


def test_scalar_axis_is_skipped():
    """A scalar 'wavelength' has no tensor dim: reducing it must be a no-op,
    so one detector config works for both mono- and polychromatic grids."""
    M = 3
    sp = make_sim_params(mode=M)  # scalar wavelength
    detector = Detector(sp, func="intensity", reduce_axes=("mode", "wavelength"))
    field = torch.randn(M, H, W, dtype=torch.complex64)
    out = detector.forward(Wavefront(field))
    assert out.shape == (H, W)


def test_reduce_mode_and_vector_wavelength():
    """With a vector wavelength axis both ensemble axes are summed out."""
    M, Nwl = 3, 4
    sp = make_sim_params(mode=M, wavelength=torch.linspace(0.5e-6, 0.7e-6, Nwl))
    # layout: (M, Nwl, H, W) — mode appended last => outermost
    field = torch.randn(M, Nwl, H, W, dtype=torch.complex64)
    detector = Detector(sp, func="intensity", reduce_axes=("mode", "wavelength"))
    out = detector.forward(Wavefront(field))
    assert out.shape == (H, W)
    assert torch.allclose(out, field.abs().pow(2).sum(dim=(0, 1)), atol=1e-5)


def test_unknown_axis_raises():
    with pytest.raises(AxisNotFound):
        Detector(make_sim_params(), func="intensity", reduce_axes=("mode",))


def test_reduce_matches_pcw_detect():
    """Cross-check: folded stack + Detector == PartiallyCoherentWavefront.detect."""
    coherence = pytest.importorskip("svetlanna.coherence")
    from svetlanna.coherence.wavefront_pc import _make_sim_params_modes

    M = 4
    sp = make_sim_params()
    sp_modes = _make_sim_params_modes(sp, M)

    phi = torch.randn(M, H, W, dtype=torch.complex64)
    w = torch.rand(M)
    pcw = coherence.PartiallyCoherentWavefront(Wavefront(phi), w, sp, sp_modes)
    expected = pcw.detect(nn.Identity())

    folded = Wavefront(w.sqrt()[:, None, None] * phi)
    detector = Detector(sp_modes, func="intensity", reduce_axes=("mode",))
    out = detector.forward(folded)

    assert torch.allclose(out, expected, atol=1e-6)


def test_detector_gradient_flows_through_reduction():
    """Gradients must reach the (complex) field through the reduced intensity."""
    M = 3
    sp = make_sim_params(mode=M)
    field = torch.randn(M, H, W, dtype=torch.complex64, requires_grad=True)
    detector = Detector(sp, func="intensity", reduce_axes=("mode",))
    out = detector.forward(Wavefront(field))
    out.sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


def test_processor_forward_keeps_autograd():
    """DetectorProcessorClf.forward must not break the autograd graph."""
    num_classes = 4
    sp = make_sim_params()
    processor = DetectorProcessorClf(num_classes=num_classes, simulation_parameters=sp)

    detector_data = torch.rand(H, W, requires_grad=True)
    probs = processor.forward(detector_data)

    assert probs.shape == (1, num_classes)
    assert probs.requires_grad
    probs.sum().backward()
    assert detector_data.grad is not None
    # probabilities still normalized
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
