# Architecture of the `svetlanna.coherence` module

This document explains the **structure and logic** of the code that models partial
coherence. For the optics behind it, see [PHYSICS.md](PHYSICS.md).

---

## 1. Design philosophy: one data structure, three representations

The core observation (derived in [PHYSICS.md §7](PHYSICS.md)) is that coherent-mode
decomposition, source-point propagation, and Monte-Carlo sampling are all the **same data
structure**:

> a **stack of `Wavefront`s along a leading "ensemble" axis** + a **weight vector** +
> a **reduction rule** (`sum` for modes/points, `mean` for Monte-Carlo).

Everything in the module is organized around this:

```
 Source  ──▶  ensemble stack (Wavefront[..., M, H, W] + weights)
                       │
                       ▼  propagation  (one batched pass through the optical setup;
                       │               nonlinear layers walked element-by-element)
                       ▼
 Detector ──▶ reduce stack to intensity  I = Σ wₙ |S[φₙ]|²   (chunk-accumulable)
```

The optical **propagation engine is shared** with the rest of SVETlANNa
(`FreeSpace`, `DiffractiveLayer`, …); coherence only changes *what* enters the stack and
*how* it is reduced — never the propagation kernel.

---

## 2. Module map

Current files (all under `svetlanna/coherence/`):

| File | Role |
|---|---|
| [`wavefront_pc.py`](wavefront_pc.py) | `PartiallyCoherentWavefront` — the coherent-mode field container + propagation/detection. **The central object.** |
| [`sources.py`](sources.py) | Physical source constructors that build the mode stack (GSM, Schell, eigendecomposition, Filipovich). |
| [`sampling.py`](sampling.py) | Monte-Carlo path: random realizations + averaging detection. |
| [`gamma.py`](gamma.py) | Dense $\Gamma$ reference (`SpatialCoherence` + `coherence_forward`) for cross-validation on small grids. |
| [`assess.py`](assess.py) | Read-only coherence-regime + cost + GPU-memory diagnostic. |
| [`__init__.py`](__init__.py) | Public exports. |

Planned (see §8): rename to `CoherentWavefrontStack`; add a `Source` front-end, a
`CoherenceDetector`, and a `backends/` package.

---

## 3. `PartiallyCoherentWavefront` — the central container

Defined in [wavefront_pc.py](wavefront_pc.py). Represents
$\Gamma=\sum_n\lambda_n\phi_n^*\phi_n$ via the coherent modes.

### 3.1 Data model
Four attributes:
- `modes : Wavefront` — complex amplitudes, shape `(..., M, H, W)`. The `M` ("mode") axis is
  the ensemble axis.
- `weights : torch.Tensor` — eigenvalues $\lambda_n\ge0$, shape `(M,)`; $\sum_n\lambda_n$ = total power.
- `sim_params : SimulationParameters` — the **spatial** grid optical elements use (no mode axis).
- `sim_params_modes : SimulationParameters` — `sim_params` extended with a `mode` axis of size `M`.

### 3.2 The mode-axis convention (important)
SVETlANNa addresses tensor axes by **negative index** via `sim_params.index(name)`. The helper
`_make_sim_params_modes()` appends `mode` last, which (by SVETlANNa's "later-inserted axes get
smaller negative indices" rule) makes `mode` the **outermost** non-scalar axis. All methods read
the mode position with `sim_params_modes.index("mode")` and broadcast weights with
`_weights_broadcast(weights, ndim, mode_dim)`. This is why elements — which operate on the
trailing spatial axes and treat leading axes as batch — propagate the whole mode stack unchanged.

### 3.3 Key methods
- `intensity` → $\sum_n\lambda_n|\phi_n|^2$ (incoherent superposition).
- `total_power` → $\sum_n\lambda_n$.
- `through(network)` → propagate all modes in one batched pass; weights unchanged. **Exact for any
  linear network** (`Γ' = UΓU†` in mode form).
- `detect(network)` → $\sum_n\lambda_n|network(\phi_n)|^2$ with the mode axis summed out. The
  training-facing entry point: a single batched forward with gradients flowing to all parameters.
- `truncate(power_fraction)` → keep the fewest top modes capturing `power_fraction` of power.
- `reorthogonalize()` → restore canonical Mercer form via SVD of $\sqrt\lambda\,\Phi$; needed after
  **lossy** elements (apertures/absorption) that break mode orthogonality. No-op for pure phase
  systems (unitary).
- `degree_of_coherence(idx1, idx2)` → $\mu(\mathbf r_1,\mathbf r_2)$ from the stored modes.
- `coherence_matrix(force=False)` → reconstruct full $\Gamma$ as `(H,W,H,W)` (debug; capped at $64^2$).
- `sample_realizations(k)` → bridge to the Monte-Carlo path (delegates to `sampling`).

> **Limitation:** `detect()`/`through()` apply the network to each mode *independently*, which is
> correct only for **linear** stacks. Nonlinear layers need the element-by-element shared-response
> walk (currently the Monte-Carlo path; see §8 for folding it into the modal path).

---

## 4. `sources.py` — building the mode stack

Four constructors, all returning a `PartiallyCoherentWavefront` (or, for Filipovich, an object
that produces one per image):

- **`gaussian_schell_model(sim_params, waist_radius, coherence_width, power, power_fraction, max_modes)`**
  Analytic Starikov–Wolf decomposition of a Gaussian Schell-model beam. Computes
  $a=1/w_0^2$, $b=1/(2\sigma_c^2)$, $c=\sqrt{a^2+2ab}$, $q=b/(a+b+c)$, mode waist $\sqrt{2/c}$;
  1-D eigenvalues $\lambda_n=(1-q)q^n$, 2-D by tensor product; builds Hermite–Gauss modes
  (`Wavefront.hermite_gauss`) and L2-normalizes each on the grid. Raises if `coherence_width ≤ 0`
  (use the incoherent sampler instead).

- **`modes_from_coherence_matrix(gamma, sim_params, power_fraction)`**
  Generic path: reshape `Γ (H,W,H,W) → (N,N)`, symmetrize, `torch.linalg.eigh`, clip negative
  numerical noise, sort descending, truncate to `power_fraction`. $O(N^3)$ — practical for $N\le64^2$.

- **`schell_model(sim_params, intensity_profile, mu_profile, build_grid, power_fraction)`**
  General Schell-model: builds $\Gamma=\sqrt{I_1I_2}\,\mu(\mathbf r_1-\mathbf r_2)$ on a coarse
  `build_grid×build_grid` mesh, eigendecomposes it, then **interpolates** each mode up to the full
  grid and renormalizes.

- **`FilipoviCoherenceSource(mu, src_shape, sim_params, power_fraction, max_modes)`**
  The DONN-training-oriented model. In `__init__` it builds the **image-independent** matrix
  $C_\mu(m,m')=\mu^{\lVert\mathbf r_m-\mathbf r_{m'}\rVert}$ on the source grid and eigendecomposes
  it **once**; eigenvectors are upsampled to the propagation grid. `from_image(intensity)` then
  produces per-image modes $\phi_n=\sqrt I\odot v_n$ cheaply (one elementwise multiply, no eigensolve);
  `from_image_batch` stacks a whole data batch on a leading axis for one GPU pass.

---

## 5. `sampling.py` — the Monte-Carlo path

- **`sample_realizations(pcw, k)`** → draw `k` fields $\psi^{(i)}=\sum_n\sqrt{\lambda_n}c_n^{(i)}\phi_n$,
  $c_n\sim\mathcal{CN}(0,1)$. Returns a `Wavefront (k, …, H, W)` whose `|·|²` averages to `pcw.intensity`.
- **`incoherent_source(sim_params, intensity, k)`** → δ-correlated source realizations
  $\psi=\sqrt I\,e^{i\theta}$, $\theta\sim\mathrm U(0,2\pi)$ per pixel. This is the
  "modal-expansion" baseline used by prior incoherent-DONN work.
- **`detect_realizations(realizations, network, weights=None)`** → propagate the realization batch
  and average $|network(\psi)|^2$ over the leading axis. **The only rigorous path when the network
  contains nonlinear elements.** Statistical error $\sim K^{-1/2}$.

---

## 6. `gamma.py` — dense reference

- **`SpatialCoherence(data, sim_params)`** holds $\Gamma$ as `(..., H, W, H, W)`, capped at $64^2$.
  `.intensity` is the diagonal; `.from_pcw()` builds it from a `PartiallyCoherentWavefront`.
- **`coherence_forward(element, gamma)`** computes $\Gamma'=U\Gamma U^\dagger$ using the identity
  $U\Gamma U^\dagger=(U(U\Gamma)^\dagger)^\dagger$ — two applications of `element.forward()` on the
  trailing `(y2,x2)` axes plus Hermitian adjoints, so no explicit adjoint operator is needed.

This module exists purely to **cross-validate** the efficient mode/MC paths on small grids.

---

## 7. `assess.py` — the regime & cost diagnostic

A read-only tool (allocates nothing on device). Three public symbols:

### `SourceSpec` (dataclass)
Describes the illumination for assessment:
- spatial coherence via `coherence_length` (l_c) **or** physical `source_size` + `source_distance`
  (→ vCZ $l_c=\lambda d/a$ through `resolve_coherence_length`);
- `n_wavelengths` (spectral channels; defaults to the sim wavelength-axis length);
- `n_source_points` (emitter count for the source-point estimate; defaults to the full grid as a
  worst case — set to the nonzero-pixel count for a realistic sparse-object estimate).

### `assess(setup=None, source, *, sim_params=None, distance=None, dtype, memory_budget_bytes=None, verbose=True)`
Logic:
1. **Geometry** (`_gather_geometry`): from a `LinearOpticalSetup`, read `sim_params` from the first
   element and take $d_i$ = the largest `FreeSpace.distance` (resolution-limiting hop); or accept
   `sim_params` + `distance` directly. Aperture $D=\min$ extent of `x,y`; pixel and `n_pixels` from
   the grid; $\lambda_{\text{center}}$ from `sim_params.wavelength` (scalar or array).
2. **Coherence**: $S_F=2\lambda d_i/D$, incoherent threshold $S_F/\sqrt\pi$, $l_c$ from the source,
   ratio $l_c/S_F$. Regime: **coherent** ($l_c\ge D$) / **incoherent** ($l_c\le S_F/\sqrt\pi$) / **partial**.
3. **Ensemble & representation** (decision tree):
   - coherent → `1`, "single coherent Wavefront";
   - else estimate modes $M\approx\lceil D/l_c\rceil^2$ (clamped to `n_pixels`);
     - partial **and** $M\le$ `_MODE_BUDGET` (256) → "modes (Mercer)";
     - else → "source-point / Monte-Carlo" (ensemble = `n_pixels` for incoherent, or
       `n_source_points` if supplied).
   - propagations/state = ensemble × `n_wavelengths`.
4. **Memory**: bytes/field = `n_pixels × itemsize(dtype)`; eager stack = ensemble × n_λ × bytes/field.
   Budget = `memory_budget_bytes`, else **free CUDA memory** via `torch.cuda.mem_get_info` when the
   grid is on GPU. `fits_eager = eager × _EAGER_OVERHEAD (3×) ≤ budget`. If it does **not** fit,
   `warnings.warn` with a concrete recommendation (chunk / coarsen / fewer λ / higher coherence).
5. Returns a **`CoherenceAssessment`** dataclass (printed by `verbose`, ASCII-only `report()` for
   cp1251/Windows consoles).

---

## 8. Planned refactor (roadmap)

Agreed direction (decisions: no polarization for now; rename; chunkable reduction):

1. **`assess.py`** — done.
2. **Rename** `PartiallyCoherentWavefront → CoherentWavefrontStack`; make `reduction`
   (`"sum"`/`"mean"`) and an `iter_chunks(size)` iterator explicit so the ensemble axis can mean
   modes / points / realizations interchangeably, and reduction is **memory-bounded**.
3. **`Source` front-end** unifying §4 constructors as **backends** (`backends/`), exposing
   `realize(representation="modes"|"points"|"montecarlo", chunk=None)` and carrying shape, pixel,
   spatial/temporal coherence, spectral range (polarization reserved, deferred).
4. **Source-point representation** for the incoherent regime (sparse, nonzero-only, vCZ via a
   source→object `FreeSpace(d)`).
5. **`CoherenceDetector`** accepting the stack, **accumulating** intensity in chunks, computing
   statistics (per-class regions, variance/SNR, degree of coherence), and hosting the
   **nonlinear-aware stepwise walk** (folds nonlinear support into all representations).
6. **Spectral/temporal axis**: wavelength as a second stacking axis; $\varphi\propto1/\lambda$ masks.

Backward compatibility: keep old names re-exported from [`__init__.py`](__init__.py) during the rename.

---

## 9. Typical data flow

```python
import torch, svetlanna as sv
from svetlanna.coherence import assess, SourceSpec, FilipoviCoherenceSource

sim = sv.SimulationParameters(x=..., y=..., wavelength=...)
setup = sv.LinearOpticalSetup(elements=[...])           # masks + FreeSpace

# 1) Diagnose the regime and whether eager realization fits.
assess(setup, SourceSpec(source_size=50e-6, source_distance=0.05))

# 2) Build the partially-coherent source (mode representation).
src = FilipoviCoherenceSource(mu=0.9, src_shape=(28, 28), sim_params=sim)
pcw = src.from_image(intensity)        # PartiallyCoherentWavefront

# 3) Propagate + detect in one batched pass (linear stack).
I_out = pcw.detect(setup)              # Σ λₙ |setup(φₙ)|²
```

For nonlinear stacks today, use `sampling.incoherent_source` / `sample_realizations` +
`detect_realizations`; after the refactor the `CoherenceDetector` will handle nonlinear
layers within any representation.

---

## 10. Cross-checks (how the pieces validate each other)

- **Dense vs modal/MC** (small grid): `gamma.coherence_forward` propagating a $\Gamma$ built by
  `SpatialCoherence.from_pcw` must match `pcw.detect()` to numerical precision.
- **Modes vs Monte-Carlo**: `detect_realizations` converges to `detect()` as $K\to\infty$.
- **`assess` vs hand calculation**: the regime verdict and mode estimate match the analytic
  $l_c=\lambda d/a$, $S_F=2\lambda d_i/D$ values (verified for the VCSEL / microLED / LED cases).
- **Chunked vs eager** (post-refactor): accumulated reduction equals the single-batch reduction.
