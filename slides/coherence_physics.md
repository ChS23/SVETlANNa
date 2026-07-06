# Coherence physics in SVETlANNa — in brief

**Coherence** = how correlated the optical field is between two points (spatial) and two
times (temporal). It is captured by the **mutual coherence function**

$$\Gamma(\mathbf r_1,\mathbf r_2)=\langle E(\mathbf r_1)\,E^*(\mathbf r_2)\rangle .$$

- **Spatial coherence length** `l_c` — how far apart two points still interfere
  (van Cittert–Zernike: `l_c = λd/a`, source size `a`, distance `d`).
- **Temporal coherence length** `L_c ≈ λ²/Δλ` — set by the spectral width.

A detector measures only **intensity** = the diagonal `I(r) = Γ(r,r)`.

---

# How SVETlANNa computes it

Any partially coherent field is an **incoherent sum of coherent modes** (Mercer/Wolf):

$$\Gamma=\sum_n \lambda_n\,\phi_n^*\phi_n .$$

So the recipe is always the same:

1. **Decompose** the source into an ensemble — coherent modes, source points, or random realizations.
2. **Propagate each member coherently** through the optics (angular-spectrum / Rayleigh–Sommerfeld).
3. **Sum the intensities** at the detector: `I = Σ wₙ |S[φₙ]|²`.

Coherence only changes *what* enters the ensemble and *how many* members there are —
**not** the propagation kernel.

---

# Why it matters & which regime you're in

The regime is set by `l_c` vs the system's finest feature `S_F = 2λd/D`:

| `l_c` vs `S_F` | regime | # ensemble members | typical source |
|---|---|---|---|
| `l_c ≫ D` | coherent | 1 | single-mode laser / VCSEL |
| `l_c ~ S_F` | partial | `≈ (D/l_c)²` modes | microLED |
| `l_c ≲ S_F` | incoherent | `≈ N` pixels | bare LED |

- **Imaging** systems are nearly coherence-insensitive; **diffractive** systems are **not** —
  the output intensity changes qualitatively with `l_c`.
- Fully incoherent light is **full-rank** (no compression) → cost ≈ N propagations: the intrinsic floor.
- `coherence.assess(setup, source)` reports the regime, the ensemble size, and the memory cost.
