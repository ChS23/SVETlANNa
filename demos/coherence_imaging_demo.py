"""Coherence demo: an MNIST digit through a random-phase DOE onto a detector.

Reproduces the qualitative effect of Kleiner et al. (arXiv:2408.06681, Fig. 1):
for a *non-imaging* element (a random phase diffuser) the detector intensity
depends dramatically on the spatial coherence of the illumination — unlike a
lens, which would image the digit nearly identically in all three cases.

Three sources are compared, with coherence lengths chosen relative to the
system's minimal feature S_F so they land in distinct regimes:
    VCSEL     -> coherent    (speckle)
    microLED  -> partial     (intermediate)
    bare LED  -> incoherent  (smooth)

Each partially coherent field is synthesized as a Schell-model Monte-Carlo
ensemble:  psi_k = sqrt(I_object) * u_k, where u_k is a complex Gaussian random
field with correlation length l_c (white noise low-pass filtered by a Gaussian).
The detector intensity is the ensemble average of |setup(psi_k)|^2, computed via
svetlanna.coherence.detect_realizations.

Run:  python demos/coherence_imaging_demo.py
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import svetlanna as sv
from svetlanna import SimulationParameters, Wavefront
from svetlanna.elements import FreeSpace, DiffractiveLayer
from svetlanna.coherence import assess, SourceSpec, detect_realizations

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))

# ── geometry (picked so the three regimes are visibly different) ───────────
N = 128                      # grid points per side
PIXEL = 10e-6                # 10 um
APERTURE = N * PIXEL         # 1.28 mm
WAVELENGTH = 550e-9          # 550 nm
DISTANCE = 5e-2              # 5 cm object->DOE and DOE->detector
K = 200                      # Monte-Carlo realizations

S_F = 2 * WAVELENGTH * DISTANCE / APERTURE      # minimal feature, ~ 43 um
print(f"minimal feature S_F = {S_F*1e6:.1f} um  ({S_F/PIXEL:.1f} px)\n")

# coherence length of each source (chosen relative to S_F and the aperture)
SOURCES = {
    "VCSEL (coherent)":   3.0e-3,    # >> aperture
    "microLED (partial)": 120e-6,    # a few * S_F
    "bare LED (incoherent)": 10e-6,  # ~ pixel, < S_F
}


def load_digit(side: int = 96) -> torch.Tensor:
    """Return a (side, side) MNIST digit in [0,1]; fall back to a drawn '5'."""
    try:
        from torchvision import datasets
        ds = datasets.MNIST(root=os.path.join(HERE, "_mnist"),
                            train=True, download=True)
        img, _ = ds[7]                       # a clean sample
        arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
        arr = arr.reshape(img.size[1], img.size[0]) / 255.0
        print("loaded MNIST digit")
    except Exception as e:                   # offline / no torchvision
        print(f"MNIST unavailable ({type(e).__name__}); using a synthetic '5'")
        pat = [
            "01111110", "01100000", "01100000", "01111100", "00000110",
            "00000011", "00000011", "01000110", "00111100", "00000000",
        ]
        arr = torch.tensor([[float(c) for c in row] for row in pat])
    arr = arr[None, None]
    arr = F.interpolate(arr, size=(side, side), mode="bilinear", align_corners=False)
    return arr[0, 0].clamp(0, 1)


def object_intensity() -> torch.Tensor:
    """Place the digit (as intensity) centered on the N×N grid."""
    side = 96
    digit = load_digit(side)
    I = torch.zeros(N, N)
    off = (N - side) // 2
    I[off:off + side, off:off + side] = digit
    return I.to(DEVICE)


def build_setup() -> sv.LinearOpticalSetup:
    sim = SimulationParameters(
        x=torch.linspace(-APERTURE / 2, APERTURE / 2, N),
        y=torch.linspace(-APERTURE / 2, APERTURE / 2, N),
        wavelength=WAVELENGTH,
    ).to(DEVICE)
    # random phase DOE (diffuser): phase uniform in [0, 2pi)
    mask = (torch.rand(N, N, generator=torch.Generator().manual_seed(1)) * 2 * math.pi).to(DEVICE)
    fs = lambda: FreeSpace(sim, distance=DISTANCE, method="zpASM",
                           total_paddings_x=N, total_paddings_y=N)
    setup = sv.LinearOpticalSetup(elements=[fs(), DiffractiveLayer(sim, mask), fs()])
    return setup, sim


def realize(I_obj: torch.Tensor, l_c: float, k: int, gen: torch.Generator) -> Wavefront:
    """k Schell-model realizations with coherence length l_c (meters).

    psi = sqrt(I) * u,  u = (complex white noise) low-pass filtered to corr. length l_c.
    l_c -> 0  recovers a delta-correlated (incoherent) source;
    l_c -> inf recovers a single coherent field (random global phasor).
    """
    s_px = l_c / PIXEL                                   # correlation length in pixels
    white = (torch.randn(k, N, N, generator=gen) + 1j * torch.randn(k, N, N, generator=gen))
    white = (white / math.sqrt(2)).to(DEVICE)
    fy = torch.fft.fftfreq(N, device=DEVICE).reshape(N, 1)
    fx = torch.fft.fftfreq(N, device=DEVICE).reshape(1, N)
    Hf = torch.exp(-2 * math.pi**2 * s_px**2 * (fx**2 + fy**2))   # Gaussian low-pass
    u = torch.fft.ifft2(torch.fft.fft2(white) * Hf)
    u = u / u.abs().pow(2).mean().sqrt()                 # normalize <|u|^2> = 1
    psi = I_obj.sqrt().unsqueeze(0) * u
    return Wavefront(psi.to(torch.complex64))


def centered_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a - a.mean()).flatten()
    b = (b - b.mean()).flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    I_obj = object_intensity()
    setup, sim = build_setup()
    gen = torch.Generator().manual_seed(42)

    outputs: dict[str, torch.Tensor] = {}
    for name, l_c in SOURCES.items():
        print(f"=== {name}: l_c = {l_c*1e6:.1f} um ===")
        assess(setup, SourceSpec(coherence_length=l_c, name=name), verbose=True)
        real = realize(I_obj, l_c, K, gen)
        with torch.no_grad():
            outputs[name] = detect_realizations(real, setup).cpu()
        print()

    # qualitative comparison vs the coherent case
    coh = outputs["VCSEL (coherent)"]
    print("centered-cosine similarity to the COHERENT output:")
    for name, I in outputs.items():
        print(f"  {name:24s}: {centered_cosine(coh, I):.3f}")

    # ── figure ──
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(I_obj.cpu(), cmap="gray"); axes[0].set_title("object (digit)")
    for ax, (name, I) in zip(axes[1:], outputs.items()):
        ax.imshow(I, cmap="inferno")
        ax.set_title(f"{name}\nl_c={SOURCES[name]*1e6:.0f} um")
    for ax in axes:
        ax.axis("off")
    out_png = os.path.join(HERE, "coherence_imaging_demo.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
