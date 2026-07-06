"""Coherence regime assessment for diffractive optical systems.

Given an optical setup (or a bare simulation grid) and a description of the
illumination source, ``assess`` reports which coherence regime the system
operates in, how many ensemble entries (coherent modes / source points /
random realizations) a faithful simulation needs, which representation to use,
and whether an **eager** realization of the whole ensemble fits in GPU memory.

The physics follows the van Cittert--Zernike relation and the incoherent-regime
criterion of Kleiner, Michaeli & Michaeli (arXiv:2408.06681):

    minimal feature preserved by the system   S_F  = 2 λ d_i / D
    spatial coherence length on the object    l_c  = λ d / a        (square source)
    incoherent approximation valid when       l_c ≲ S_F / sqrt(π)

where D is the aperture (grid extent), d_i the object-to-element propagation
distance, a the source side length and d the source-to-object distance.

This module is **read-only / diagnostic** — it allocates nothing on the device.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

from svetlanna.simulation_parameters import SimulationParameters
from svetlanna.elements.free_space import FreeSpace
from svetlanna.setup import LinearOpticalSetup

# Multiplier on the bare stack size to account for FFT temporaries, the
# propagation transfer function, and (during training) autograd activations.
_EAGER_OVERHEAD = 3.0

# Above this many coherent modes the Mercer representation stops paying off and
# a source-point / Monte-Carlo representation is recommended instead.
_MODE_BUDGET = 256


@dataclass
class SourceSpec:
    """Lightweight description of an illumination source for assessment.

    Provide spatial coherence either directly via ``coherence_length`` (l_c) or
    physically via ``source_size`` (a) and ``source_distance`` (d); in the
    latter case l_c = λ·d/a is computed from the simulation wavelength.

    Parameters
    ----------
    coherence_length : float | None
        Spatial coherence length l_c on the object plane (same length units as
        the simulation grid). Takes precedence over ``source_size``/``distance``.
    source_size : float | None
        Source side length a (square source), for the van Cittert--Zernike estimate.
    source_distance : float | None
        Source-to-object distance d.
    n_wavelengths : int | None
        Number of spectral channels to simulate. If None, taken from the
        simulation wavelength axis (1 if monochromatic).
    n_source_points : int | None
        Number of independent emitters for the source-point representation
        (e.g. the count of nonzero object pixels). If None, the full grid is
        used as a worst-case upper bound. Ignored unless the recommended
        representation is source-point / Monte-Carlo.
    name : str
        Optional label for the report.
    """

    coherence_length: float | None = None
    source_size: float | None = None
    source_distance: float | None = None
    n_wavelengths: int | None = None
    n_source_points: int | None = None
    name: str = "source"

    def resolve_coherence_length(self, wavelength_center: float) -> float:
        if self.coherence_length is not None:
            return float(self.coherence_length)
        if self.source_size is not None and self.source_distance is not None:
            if self.source_size <= 0:
                raise ValueError("source_size must be > 0")
            return wavelength_center * self.source_distance / self.source_size
        raise ValueError(
            "SourceSpec needs either coherence_length, or both source_size and "
            "source_distance, to determine the spatial coherence length."
        )


@dataclass
class CoherenceAssessment:
    """Result of :func:`assess` — numbers plus a human-readable ``report``."""

    # geometry
    aperture: float
    pixel: float
    n_pixels: int
    wavelength_center: float
    n_wavelengths: int
    distance: float
    # coherence
    feature_size: float          # S_F
    incoherent_threshold: float  # S_F / sqrt(pi)
    coherence_length: float      # l_c
    coherence_to_feature: float  # l_c / S_F
    regime: str                  # "coherent" | "partial" | "incoherent"
    # cost
    representation: str
    ensemble_size: int           # modes M, or source-point count
    n_propagations: int          # ensemble_size * n_wavelengths
    # memory
    eager_bytes: int
    gpu_free_bytes: int | None
    gpu_total_bytes: int | None
    fits_eager: bool | None

    def report(self) -> str:
        def _b(n: int | None) -> str:
            if n is None:
                return "n/a"
            return f"{n / 1024**3:.2f} GiB"

        lines = [
            "-- Coherence assessment ------------------------------------",
            f"  grid            : {self.n_pixels} px, pixel {self.pixel:.3g}, "
            f"aperture {self.aperture:.3g}",
            f"  wavelength      : {self.wavelength_center:.3g} "
            f"(x {self.n_wavelengths} spectral channel(s))",
            f"  propagation d_i : {self.distance:.3g}",
            f"  feature S_F     : {self.feature_size:.3g}  "
            f"(incoherent if l_c <~ {self.incoherent_threshold:.3g})",
            f"  coherence l_c   : {self.coherence_length:.3g}  "
            f"(l_c / S_F = {self.coherence_to_feature:.3g})",
            f"  REGIME          : {self.regime.upper()}",
            f"  representation  : {self.representation}",
            f"  ensemble        : {self.ensemble_size} entries x "
            f"{self.n_wavelengths} wl = {self.n_propagations} propagations/state",
            f"  eager memory    : {_b(self.eager_bytes)} "
            f"(stack, x{_EAGER_OVERHEAD:g} runtime overhead)",
            f"  gpu free / total: {_b(self.gpu_free_bytes)} / {_b(self.gpu_total_bytes)}",
            f"  fits eager?     : {self.fits_eager}",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _gather_geometry(
    setup: LinearOpticalSetup | None,
    sim_params: SimulationParameters | None,
    distance: float | None,
) -> tuple[SimulationParameters, float]:
    """Return (sim_params, d_i) from a setup or explicit arguments.

    d_i is taken as the *largest* free-space propagation distance in the setup
    (the resolution-limiting hop for the cascade); override with ``distance``.
    """
    if setup is not None:
        if not setup.elements:
            raise ValueError("setup has no elements")
        sp = setup.elements[0].simulation_parameters
        distances = [
            _as_float(el.distance) for el in setup.elements if isinstance(el, FreeSpace)
        ]
        if distance is not None:
            d_i = float(distance)
        elif distances:
            d_i = max(distances)
        else:
            raise ValueError(
                "setup contains no FreeSpace element; pass distance=... explicitly."
            )
        return sp, d_i

    if sim_params is None or distance is None:
        raise ValueError("Without a setup, both sim_params and distance are required.")
    return sim_params, float(distance)


def assess(
    setup: LinearOpticalSetup | None = None,
    source: SourceSpec | None = None,
    *,
    sim_params: SimulationParameters | None = None,
    distance: float | None = None,
    dtype: torch.dtype = torch.complex64,
    memory_budget_bytes: int | None = None,
    verbose: bool = True,
) -> CoherenceAssessment:
    """Assess the coherence regime and simulation cost of an optical system.

    Parameters
    ----------
    setup : LinearOpticalSetup | None
        Optical setup; geometry (aperture, propagation distance) is read from it.
        If None, pass ``sim_params`` and ``distance`` explicitly.
    source : SourceSpec
        Illumination description (spatial/temporal coherence).
    sim_params : SimulationParameters | None
        Simulation grid, when no setup is given.
    distance : float | None
        Object-to-element distance d_i; overrides the value read from the setup.
    dtype : torch.dtype
        Complex dtype the ensemble would be stored in (memory estimate).
    memory_budget_bytes : int | None
        Manual memory budget for the eager-fit check (e.g. on CPU, where GPU
        free memory is unavailable). If None and the grid is on CUDA, the free
        GPU memory is queried.
    verbose : bool
        Print the report.

    Returns
    -------
    CoherenceAssessment

    Warns
    -----
    UserWarning
        If an eager realization of the full ensemble would not fit in the
        available (or budgeted) memory — recommends chunked realization.
    """
    if source is None:
        raise ValueError("a SourceSpec must be provided")

    sp, d_i = _gather_geometry(setup, sim_params, distance)

    # --- geometry ---
    x, y = sp.x, sp.y
    nx, ny = len(x), len(y)
    lx = abs(_as_float(x[-1]) - _as_float(x[0]))
    ly = abs(_as_float(y[-1]) - _as_float(y[0]))
    aperture = min(lx, ly)
    pixel = lx / max(nx - 1, 1)
    n_pixels = nx * ny

    wl = sp.wavelength
    wl_t = wl if wl.dim() > 0 else wl.reshape(1)
    wavelength_center = float(wl_t.to(torch.float64).mean().item())
    n_wavelengths = (
        int(source.n_wavelengths)
        if source.n_wavelengths is not None
        else (len(wl_t) if wl.dim() > 0 else 1)
    )

    # --- coherence ---
    feature_size = 2.0 * wavelength_center * d_i / aperture          # S_F
    incoherent_threshold = feature_size / math.sqrt(math.pi)
    l_c = source.resolve_coherence_length(wavelength_center)
    ratio = l_c / feature_size if feature_size > 0 else math.inf

    if l_c >= aperture:
        regime = "coherent"
    elif l_c <= incoherent_threshold:
        regime = "incoherent"
    else:
        regime = "partial"

    # --- ensemble size & representation ---
    if regime == "coherent":
        ensemble = 1
        representation = "single coherent Wavefront"
    else:
        # number of coherence cells across the aperture, per dimension
        cells_1d = max(1.0, aperture / max(l_c, pixel))
        ensemble = min(n_pixels, int(math.ceil(cells_1d) ** 2))
        if regime == "partial" and ensemble <= _MODE_BUDGET:
            representation = "modes (Mercer / CoherentWavefrontStack)"
        else:
            # incoherent is full rank (≈ one entry per source pixel); a partial
            # field with too many modes is also cheaper as source points.
            if regime == "incoherent":
                ensemble = n_pixels
            if source.n_source_points is not None:
                ensemble = int(source.n_source_points)
            representation = "source-point / Monte-Carlo"

    n_propagations = ensemble * n_wavelengths

    # --- memory ---
    itemsize = torch.empty(0, dtype=dtype).element_size()
    bytes_per_field = n_pixels * itemsize
    eager_bytes = int(ensemble * n_wavelengths * bytes_per_field)

    gpu_free = gpu_total = None
    if sp.device.type == "cuda":
        gpu_free, gpu_total = torch.cuda.mem_get_info(sp.device)

    budget = memory_budget_bytes if memory_budget_bytes is not None else gpu_free
    if budget is None:
        fits_eager = None
    else:
        fits_eager = eager_bytes * _EAGER_OVERHEAD <= budget

    result = CoherenceAssessment(
        aperture=aperture,
        pixel=pixel,
        n_pixels=n_pixels,
        wavelength_center=wavelength_center,
        n_wavelengths=n_wavelengths,
        distance=d_i,
        feature_size=feature_size,
        incoherent_threshold=incoherent_threshold,
        coherence_length=l_c,
        coherence_to_feature=ratio,
        regime=regime,
        representation=representation,
        ensemble_size=ensemble,
        n_propagations=n_propagations,
        eager_bytes=eager_bytes,
        gpu_free_bytes=gpu_free,
        gpu_total_bytes=gpu_total,
        fits_eager=fits_eager,
    )

    if fits_eager is False:
        warnings.warn(
            f"Eager realization of '{source.name}' needs "
            f"~{eager_bytes * _EAGER_OVERHEAD / 1024**3:.2f} GiB "
            f"({ensemble} entries x {n_wavelengths} wl x {bytes_per_field / 1024**2:.1f} MiB, "
            f"x{_EAGER_OVERHEAD:g} overhead) but only "
            f"{budget / 1024**3:.2f} GiB is available. "
            "Use chunked realization (iter_chunks) or reduce the ensemble "
            "(coarser source grid / fewer wavelengths / higher coherence).",
            UserWarning,
            stacklevel=2,
        )

    if verbose:
        print(result.report())

    return result
