# Integrating the coherence module into `main` — a staged plan

Audience: the SVETlANNa team. This document proposes **how to land the partial-coherence
work into the main library** as a sequence of small, reviewable, backward-compatible
pull requests, following the project's existing workflow ([CONTRIBUTING.md](../../CONTRIBUTING.md)).

See [PHYSICS.md](PHYSICS.md) for the optics and [ARCHITECTURE.md](ARCHITECTURE.md) for the
target design. This file is about **process and sequencing**, not physics.

---

## Guiding principles

1. **Additive first.** Land new code as a self-contained `svetlanna/coherence/` subpackage
   that does **not** touch existing public APIs. Risk to current users ≈ 0.
2. **One logical change per PR.** Each PR is independently reviewable, green on CI, and
   leaves `main` in a working state. No "big bang" merge of the whole feature branch.
3. **Backward compatible.** Renames keep deprecated aliases for ≥1 minor release with a
   `DeprecationWarning`. Reorganizations keep old import paths working via re-exports.
4. **Test-gated.** Every PR adds tests; the correctness gate is the **cross-check between
   representations** (modes ↔ source-point ↔ Monte-Carlo ↔ dense Γ).
5. **Each PR passes the full local gate** before review:
   ```bash
   poetry run black svetlanna
   poetry run mypy svetlanna
   poetry run flake8 svetlanna
   poetry run pytest
   ```

---

## Current state (the `coherence` feature branch)

Already implemented and validated on the branch:

- `wavefront_pc.py` — `PartiallyCoherentWavefront` (coherent-mode container, `detect`/`through`).
- `sources.py` — `gaussian_schell_model`, `modes_from_coherence_matrix`, `schell_model`, `FilipoviCoherenceSource`.
- `sampling.py` — Monte-Carlo path (`sample_realizations`, `incoherent_source`, `detect_realizations`).
- `gamma.py` — dense `SpatialCoherence` + `coherence_forward` reference.
- `assess.py` — regime/cost/GPU-memory diagnostic (`assess`, `SourceSpec`, `CoherenceAssessment`).
- `PHYSICS.md`, `ARCHITECTURE.md` — documentation.
- `demos/coherence_imaging_demo.py` — end-to-end MNIST-through-DOE demo.

**Recommendation:** do **not** merge the branch as one PR. Re-slice it into the sequence below.
A large single PR is hard to review and risky to roll back; the staged sequence lands the same
code with far better reviewability.

---

## PR sequence

| PR | Title | Type | API impact | Risk |
|----|-------|------|-----------|------|
| 1 | Coherence core (modes + sources + sampling + dense ref) | additive | new subpackage only | low |
| 2 | `assess` diagnostic + docs | additive | new functions | low |
| 3 | Rename → `CoherentWavefrontStack` (+ alias) | rename | deprecated alias | low |
| 4 | `Source` front-end + `backends/` | reorg | re-exports kept | low–med |
| 5 | Source-point representation | additive | new representation | med |
| 6 | `CoherenceDetector`: chunking + nonlinear walk | additive | new class | med |
| 7 | Spectral/temporal axis | feature | extends `Source`/detector | med |

### PR 1 — Coherence core
- **Files:** `svetlanna/coherence/{__init__.py,wavefront_pc.py,sources.py,sampling.py,gamma.py}`.
- **Scope:** the field container, the four source constructors, the Monte-Carlo path, the dense
  reference. Purely additive — no existing file changes.
- **Tests:** `tests/test_coherence_core.py` — mode orthogonality, `detect` linearity, GSM eigenvalue
  formula, `coherence_matrix` round-trip on a ≤64² grid; **cross-check** `detect()` vs
  `coherence_forward` (dense Γ) to numerical precision.
- **Reviewer focus:** the mode-axis convention (`sim_params_modes.index("mode")`), weight broadcasting,
  numerical guards (NaN handling in HG modes, eigenvalue clipping).

### PR 2 — `assess` diagnostic + docs
- **Files:** `assess.py`, `PHYSICS.md`, `ARCHITECTURE.md`, `__init__.py` exports.
- **Scope:** read-only regime/cost/memory diagnostic; allocates nothing on device.
- **Tests:** `tests/test_coherence_assess.py` — regime classification for coherent/partial/incoherent
  inputs; `S_F`/`l_c` formulas; the memory warning fires when `memory_budget_bytes` is too small;
  geometry extraction from a `LinearOpticalSetup`.
- **Reviewer focus:** ASCII-only report (Windows cp1251 consoles); `torch.cuda.mem_get_info` only
  queried on CUDA grids.

### PR 3 — Rename to `CoherentWavefrontStack`
- **Change:** rename `PartiallyCoherentWavefront` → `CoherentWavefrontStack`; add explicit
  `reduction: Literal["sum","mean"]` and `iter_chunks(size)`.
- **Backward compat:** keep `PartiallyCoherentWavefront = CoherentWavefrontStack` re-exported with a
  `DeprecationWarning` on instantiation; keep it in `__all__` for one release.
- **Tests:** alias still works and warns; `iter_chunks` reduction equals single-batch reduction.
- **Reviewer focus:** that `reduction` defaults preserve current `detect` semantics (`sum`).

### PR 4 — `Source` front-end + `backends/`
- **Change:** add `Source` (shape, pixel, spatial/temporal coherence, spectral range; polarization
  **reserved, not implemented**); move §PR1 constructors into `svetlanna/coherence/backends/`.
- **Backward compat:** re-export the moved constructors from `coherence/__init__.py` and their old
  module paths (thin shim modules) — existing imports keep working.
- **Tests:** `Source.realize("modes")` reproduces the direct-constructor result bit-for-bit.
- **Reviewer focus:** import-path stability; no behavioral change to the backends.

### PR 5 — Source-point representation
- **Change:** add `representation="points"` — propagate (nonzero) source pixels, sum intensities;
  optional source-plane `FreeSpace(d)` to imprint van Cittert–Zernike partial coherence.
- **Tests:** points ↔ modes ↔ Monte-Carlo agreement for a matched source; sparse (nonzero-only)
  equals dense on a small grid.
- **Reviewer focus:** the `N = source pixels, not grid` distinction; batching; gradient flow.

### PR 6 — `CoherenceDetector` (chunking + nonlinear)
- **Change:** detector that **accumulates** intensity over `iter_chunks` (bounded memory) and walks
  the setup element-by-element, applying the **shared intensity-dependent response** at each
  `NonlinearElement`. `CoherentWavefrontStack.detect` becomes a thin wrapper.
- **Tests:** chunked == eager; nonlinear stepwise path matches a Monte-Carlo reference for an
  intensity-dependent phase element.
- **Reviewer focus:** autograd correctness under chunking (gradient checkpointing if needed).

### PR 7 — Spectral / temporal axis
- **Change:** wavelength as a second stacking axis; λ-scaled mask phase (`φ ∝ 1/λ`); `Source` spectral
  params drive channel sampling; detector sums over wavelength.
- **Tests:** monochromatic limit (`n_λ=1`) reproduces PR1–6 results; intensity sums over channels.
- **Reviewer focus:** interaction with `SimulationParameters` polychromatic wavelength axis.

---

## Backward-compatibility & deprecation policy

- Renamed/moved symbols keep working for **at least one minor release** via aliases / re-exports.
- Aliases emit `DeprecationWarning` (`stacklevel=2`) and are listed in the changelog under
  *Deprecated*.
- Remove deprecated names only in a later **minor** bump, never in a patch.

---

## Testing strategy (the correctness gates)

The physics gives us free oracles — use them as the merge gate for every PR that touches propagation:

1. **Dense Γ reference** (`coherence_forward`) == modal `detect()` == source-point, on ≤64² grids.
2. **Monte-Carlo convergence:** `detect_realizations` → `detect()` as `K→∞` (assert MSE below a
   tolerance at fixed `K`, fixed seed).
3. **Analytic limits:** GSM eigenvalues `λ_n=(1-q)q^n`; `l_c→0` reproduces `incoherent_source`;
   `l_c→∞` reproduces a single coherent field.
4. **`assess` vs hand calculation:** regime + mode estimate match `l_c=λd/a`, `S_F=2λd_i/D`.

Use fixed seeds and small grids so tests are fast and deterministic.

---

## Repository hygiene (per CONTRIBUTING.md)

- **Do not commit** the demo's generated `*.png`, the downloaded `_mnist/` data, or any notebooks.
  Add to `.gitignore`:
  ```
  demos/*.png
  demos/_mnist/
  ```
- Keep the demo **script** (it's source), but its heavy deps (`torchvision`, `matplotlib`) must stay
  **optional** — do not add them to the core dependency group. Put them in a `demos`/`docs` extra in
  `pyproject.toml`, or import them lazily inside the demo only.
- numpy-style docstrings on all public functions/classes.

---

## Mechanics: how to introduce the changes to `main`

1. Branch from up-to-date `main`; rebase the existing `coherence` branch onto it.
2. **Re-slice** the branch into the PRs above (e.g. `git rebase -i` to group commits, or cherry-pick
   per-PR onto fresh topic branches `coherence/pr1-core`, `coherence/pr2-assess`, …).
3. For each topic branch: run the full local gate, open a PR into `main` with a clear description and
   the motivation, link the relevant section of `ARCHITECTURE.md`.
4. Merge **in order** (each PR builds on the previous). Keep PRs small enough to review in one sitting.
5. After PR 1–2 land, the diagnostic + core are usable on `main` immediately; the refactor PRs (3–7)
   follow without breaking anyone.

### Versioning & changelog
- New functionality → bump the **minor** version in `pyproject.toml`.
- Maintain a changelog with *Added* (each PR) and *Deprecated* (PR 3/4 aliases) sections.

### Rollback
- Because each PR is additive or alias-guarded, reverting any single PR restores a working `main`
  without affecting unrelated code. The dense-Γ reference stays as the long-term correctness anchor.

---

## Suggested first action

Open **PR 1 (Coherence core)** and **PR 2 (assess + docs)** together as the foundation — both are
purely additive and already validated by the demo and the cross-checks. The rename and the
`Source`/`Detector` refactor (PR 3+) can then proceed incrementally without pressure.
