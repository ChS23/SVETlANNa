# The physics of partial coherence in diffractive optical networks

This document explains the optics that the `svetlanna.coherence` module implements:
what coherence is, why it matters *more* for diffractive networks than for imaging,
how partially coherent light is represented and propagated, and where the
computational cost comes from. The companion document
[ARCHITECTURE.md](ARCHITECTURE.md) maps these ideas onto the code.

---

## 1. Why coherence matters here

A diffractive optical network (DONN) is a stack of phase masks separated by free space.
The output is an **intensity** pattern on a detector. Whether that intensity forms as we
expect depends critically on the **coherence** of the illuminating light — how correlated
the optical field is between different points (spatial coherence) and different times
(temporal coherence).

For ordinary **imaging** systems the degree of spatial coherence only mildly affects the
result (some blur/halos at edges). For **diffractive networks** it is dramatic: the masks
introduce *abrupt* phase changes, so the output intensity can change qualitatively between
the coherent and incoherent limits. A network trained assuming one coherence level can
fail completely at another (Kleiner, Michaeli & Michaeli, arXiv:2408.06681). Hence a
faithful simulator must model the *actual* coherence of the source.

---

## 2. The mutual coherence function

The complete second-order description of a (scalar, stationary) optical field is the
**mutual coherence function**

$$\Gamma(\mathbf r_1,\mathbf r_2,\tau)=\langle E(\mathbf r_1,t+\tau)\,E^*(\mathbf r_2,t)\rangle,$$

where $\langle\cdot\rangle$ is a time average. Two important reductions:

- **Mutual intensity** (equal time, $\tau=0$): $J(\mathbf r_1,\mathbf r_2)=\Gamma(\mathbf r_1,\mathbf r_2,0)$ — carries the *spatial* coherence.
- **Intensity**: $I(\mathbf r)=\Gamma(\mathbf r,\mathbf r,0)$ — the diagonal, what a detector measures.

The normalized **complex degree of coherence** is

$$\gamma(\mathbf r_1,\mathbf r_2,\tau)=\frac{\Gamma(\mathbf r_1,\mathbf r_2,\tau)}{\sqrt{I(\mathbf r_1)\,I(\mathbf r_2)}},\qquad |\gamma|\in[0,1],$$

with $|\gamma|=1$ fully coherent and $|\gamma|=0$ fully incoherent.

> In code, `PartiallyCoherentWavefront.degree_of_coherence()` evaluates the spatial
> $\mu(\mathbf r_1,\mathbf r_2)=\gamma(\mathbf r_1,\mathbf r_2,0)$ from the stored modes, and
> `coherence_matrix()` reconstructs the full $\Gamma$ on small grids.

---

## 3. Temporal coherence

Temporal coherence is set by the **spectral width** $\Delta\lambda$ (or $\Delta\nu$) of the
source. The **coherence time** and **coherence length** are, to order of magnitude,

$$\tau_c \sim \frac{1}{\Delta\nu},\qquad L_c=c\,\tau_c\sim\frac{\lambda^2}{\Delta\lambda}.$$

For a Gaussian spectrum (Goodman, *Statistical Optics*) the precise factor is
$\tau_c = 0.664/\Delta\nu$.

Examples:
- Laser / single-mode VCSEL: $\Delta\nu\sim$ MHz–GHz $\Rightarrow L_c\sim$ cm–m (effectively monochromatic).
- LED: $\Delta\lambda\sim 20$–$40$ nm $\Rightarrow L_c\sim$ a few–10 µm (broadband).

**Why it matters for a DONN:** light reaching one detector pixel from two mask regions
separated laterally by $\Delta x$ over a propagation distance $z$ acquires an optical path
difference $\sim \Delta x^2/2z$. The two contributions interfere only while this stays below
$L_c$. So a short $L_c$ limits the lateral region that can interfere coherently. Numerically,
temporal coherence is handled by sampling the spectrum into discrete wavelengths and
**summing intensities** over them, with wavelength-dependent mask phase
$\varphi(\lambda)=\varphi(\lambda_0)\,\lambda_0/\lambda$ (a height map produces a $\lambda$-dependent phase).

---

## 4. Spatial coherence and the van Cittert–Zernike theorem

An extended thermal/LED source is **spatially incoherent at the emitter** (coherence
$\sim\lambda$). Propagation *builds up* spatial coherence: the **van Cittert–Zernike**
theorem says that a uniformly bright incoherent source of area $A_s$ at distance $d$
produces, on the object plane, a coherence area

$$A_c=\frac{(\lambda d)^2}{A_s},\qquad l_c=\sqrt{A_c}=\frac{\lambda d}{a}\ \text{(square source of side }a).$$

So spatial coherence **grows with distance** and **shrinks with source size**. This is the
single most useful design relation: to make light more coherent, use a smaller source
farther away (at the cost of optical throughput).

---

## 5. The incoherent-regime criterion (why "incoherent" is rarely exact)

A system can be treated as *fully incoherent* only when the coherence length on the object
is smaller than the **finest feature the system preserves**. For an aperture (element side)
$D$ at object–element distance $d_i$, the maximal spatial frequency is
$f^{\text{sys}}_{\max}=D/(2\lambda d_i)$, so the minimal feature is

$$S_F=\frac{1}{f^{\text{sys}}_{\max}}=\frac{2\lambda d_i}{D}.$$

The incoherent approximation holds (Kleiner et al., Eq. 1) only when

$$l_c\ \lesssim\ \frac{1}{\sqrt{\pi}}\,\frac{1}{f^{\text{sys}}_{\max}+f^{\text{obj}}_{\max}}\ \approx\ \frac{S_F}{\sqrt{\pi}}.$$

Equivalently, the object emits effectively incoherent light only when the **illumination NA
is below the collection NA** ($a/d < D/d_i$). In many real settings (source near the pupil,
$a\approx D$, $d\approx d_i$) you land in the **partial** regime where *neither* the coherent
nor the incoherent approximation is valid — and you cannot ignore it.

> This criterion is exactly what `assess()` computes: it reports $S_F$, $l_c$, the ratio
> $l_c/S_F$, and classifies **coherent** ($l_c\ge D$) / **partial** / **incoherent**
> ($l_c\lesssim S_F/\sqrt\pi$).

---

## 6. Coherent-mode (Mercer / Wolf) decomposition

Because $\Gamma$ (equivalently $J$) is Hermitian and positive semi-definite, it has an
eigen-decomposition into orthonormal **coherent modes** $\phi_n$ with eigenvalues $\lambda_n\ge0$
(Wolf, *JOSA* 1982):

$$\Gamma(\mathbf r_1,\mathbf r_2)=\sum_n \lambda_n\,\phi_n^*(\mathbf r_1)\,\phi_n(\mathbf r_2).$$

Each mode is a fully coherent field that propagates **independently** through any *linear*
system; mode intensities add incoherently at the detector. This is the most compact
representation **when $\Gamma$ is low-rank**: the number of significant modes is

$$M\ \approx\ \left(\frac{\text{aperture}}{l_c}\right)^2.$$

- High coherence ($l_c$ large) $\Rightarrow M\ll N$: huge compression.
- **Fully incoherent** $\Rightarrow \Gamma(\mathbf r_1,\mathbf r_2)=I(\mathbf r_1)\delta(\mathbf r_1-\mathbf r_2)$, whose rank is $N$ (the number of source pixels). All eigenvalues are equal; **no compression is possible**.

This is the fundamental reason the modal method "does not help" for incoherent light: there
are genuinely $N$ independent degrees of freedom. It is a property of the light, not of the
algorithm.

> `gaussian_schell_model()` builds these modes analytically (Hermite–Gauss, Starikov–Wolf);
> `modes_from_coherence_matrix()` and `schell_model()` build them by eigendecomposing a given
> $\Gamma$; `FilipoviCoherenceSource` eigendecomposes the image-independent matrix
> $C_\mu(m,m')=\mu^{\lVert \mathbf r_m-\mathbf r_{m'}\rVert}$ once and reuses it per image.

---

## 7. Three equivalent representations of a partially coherent field

The same physics admits three numerical representations; all reduce to **"a set of fields
along an ensemble axis + a rule to combine intensities."**

| Representation | Ensemble member | Combine rule | Exact for linear? | Cost |
|---|---|---|---|---|
| **Coherent modes** (Mercer) | eigenmode $\phi_n$ | $\sum_n\lambda_n\lvert\cdot\rvert^2$ | yes | $M$ propagations |
| **Source points** | field of one source pixel | $\sum_j I_j\lvert\cdot\rvert^2$ | yes | $N_{\text{src}}$ propagations |
| **Monte-Carlo** | random realization | $\frac1K\sum_i\lvert\cdot\rvert^2$ | approximate ($\sim K^{-1/2}$) | $K$ propagations |

- **Source points** treat the incoherent source as independent radiators (van Cittert–Zernike
  is generated *physically* by propagating each point from the source plane). For the fully
  incoherent limit the count equals the number of (nonzero) source pixels — the same $N$ as the
  modal rank, i.e. no cheaper, but the points are **sparse and localized** so zeros can be skipped.
- **Monte-Carlo** draws random fields whose ensemble statistics reproduce $\Gamma$:
  - modal form $\psi=\sum_n\sqrt{\lambda_n}\,c_n\,\phi_n$ with $c_n\sim\mathcal{CN}(0,1)$, or
  - per-pixel form $\psi(\mathbf r)=\sqrt{I(\mathbf r)}\,e^{i\theta(\mathbf r)}$ with $\theta\sim\mathrm U(0,2\pi)$ (δ-correlated source).
  Averaging $\lvert\cdot\rvert^2$ over draws converges to the true intensity. This is the
  **random phasor sum / speckle** model (Goodman, *Speckle Phenomena in Optics*).

These are not competing theories — they are different bases/samplings of the *same* $\Gamma$.
Monte-Carlo is the only one that handles arbitrary nonlinearities directly (see §10).

---

## 8. Propagation

### 8.1 The coherent propagation kernel
Free-space propagation of a coherent field over distance $z$ uses the **angular spectrum
method** with the Rayleigh–Sommerfeld transfer function

$$H_{\text{RS}}(f_x,f_y;z,\lambda)=\exp\!\Big(i\tfrac{2\pi}{\lambda}z\sqrt{1-\lambda^2(f_x^2+f_y^2)}\Big),\quad
E(z+d)=\mathcal F^{-1}\{\mathcal F\{E\}\,H_{\text{RS}}\}.$$

This handles **arbitrary distances and geometry** and is the engine underneath every
representation (in SVETlANNa, `FreeSpace` with methods ASM/zpASM/RSC/zpRSC). A phase mask
multiplies the field by $\exp(-i\varphi(\mathbf r))$.

### 8.2 Propagating partial coherence
Two equivalent views of pushing a partially coherent field through a linear operator $U$:

1. **On $\Gamma$ directly:** $\Gamma'=U\,\Gamma\,U^\dagger$ (a four-point object, $O(N^2)$ memory). Used only as a small-grid reference.
2. **On the ensemble:** propagate every mode / point / realization with the *same* $U$, then combine intensities. This is what all production paths do — one **batched** pass over the ensemble axis.

Both give identical results for linear systems; the second is far cheaper because it never
materializes the $N\times N$ object $\Gamma$.

---

## 9. Polychromatic (temporal) propagation

Different wavelengths are mutually incoherent, so their **intensities add**:

$$I_{\text{out}}(\mathbf r)=\sum_\lambda I(\lambda)\,\big|\,\text{system}_\lambda[E_\lambda](\mathbf r)\big|^2.$$

Each wavelength sees a different propagation kernel ($H_{\text{RS}}$ depends on $\lambda$) and a
different mask phase ($\varphi\propto1/\lambda$ for a fixed height map). Numerically this is a
**second stacking axis** (wavelength) on top of the ensemble axis; the detector sums over both.
A broadband source therefore multiplies cost by the number of spectral channels $N_\lambda$.

---

## 10. Nonlinear layers

A nonlinear element responds to the **total** instantaneous intensity, so it **couples** the
ensemble members — they can no longer be propagated as fully independent channels through an
opaque stack. The correct procedure, identical in spirit for modes / points / MC, is to walk
the network layer by layer and, at each nonlinear plane:

1. form the total intensity $I_{\text{tot}}=\sum_n w_n\lvert E_n\rvert^2$ over the ensemble,
2. derive the shared intensity-dependent transmission $t=t(I_{\text{tot}})$,
3. apply the *same* $t$ to **every** ensemble member,
4. continue propagating.

This is exact for power-preserving phase nonlinearities (e.g. photorefractive media). For a
**Monte-Carlo** ensemble the nonlinearity can instead be applied to each realization directly,
because each realization is a physically valid instantaneous field — which is why MC is the
robust fallback for arbitrary nonlinearities.

---

## 11. Detection

The detector measures intensity, i.e. the diagonal of the propagated $\Gamma$:

$$I_{\text{det}}=\sum_n \lambda_n\,\big|\,S[\phi_n]\big|^2 \quad(\text{modes}),\qquad
I_{\text{det}}\approx\frac1K\sum_i \big|\,S[\psi^{(i)}]\big|^2\quad(\text{MC}),$$

where $S$ is the optical system up to the detector. Because the combination is a (weighted)
**sum**, it is associative and can be **accumulated in chunks** — the basis for bounded-memory
("lazy") evaluation when the full ensemble does not fit in memory.

---

## 12. Physical source models

- **Gaussian Schell-model (GSM):** intensity and degree of coherence are both Gaussian.
  Its modes are Hermite–Gauss functions with analytic eigenvalues $\lambda_n=(1-q)q^n$
  (Starikov–Wolf). Two parameters: beam waist $w_0$ and coherence width $\sigma_c$.
- **General Schell-model:** shift-invariant coherence $\mu(\mathbf r_1-\mathbf r_2)$ with arbitrary
  intensity $I(\mathbf r)$; $W=\sqrt{I_1 I_2}\,\mu$. Decomposed numerically on a coarse grid.
- **Filipovich exponential model:** $C_\mu(m,m')=\mu^{\lVert\mathbf r_m-\mathbf r_{m'}\rVert}$,
  $\mu\in[0,1]$ ($\mu=1$ coherent, $\mu=0$ incoherent). The coherence matrix is
  image-independent, so its eigenvectors are computed once and reused per image.
- **Fully incoherent (δ-correlated):** $\Gamma(\mathbf r_1,\mathbf r_2)=I(\mathbf r_1)\delta(\mathbf r_1-\mathbf r_2)$;
  represented by per-pixel random phase (Monte-Carlo) or per-pixel source points.

---

## 13. Cost summary

| Regime | Coherence $l_c$ | Significant modes $M$ | Cheapest representation |
|---|---|---|---|
| Coherent | $\gtrsim$ aperture | 1 | single coherent field |
| Partial | between | $\sim(\text{aperture}/l_c)^2$ | Mercer modes (if $M$ small) |
| Incoherent | $\lesssim S_F$ | $N$ (full rank) | source-point / Monte-Carlo |

Total propagations per network state $\approx (\text{ensemble size})\times N_\lambda$. The
incoherent floor — $N$ propagations — is intrinsic; the practical levers are reducing $N$
(coarser/sparser source), caching a linear intensity-transfer matrix for frozen masks, and
chunking the ensemble to stay within memory.

---

## References

- J. W. Goodman, *Statistical Optics*, 2nd ed. (Wiley, 2015).
- J. W. Goodman, *Speckle Phenomena in Optics* (Roberts & Co., 2007).
- L. Mandel & E. Wolf, *Optical Coherence and Quantum Optics* (Cambridge, 1995).
- E. Wolf, "New theory of partial coherence in the space–frequency domain. Part I," *JOSA* **72**, 343 (1982).
- M. Kleiner, L. Michaeli & T. Michaeli, "Coherence Awareness in Diffractive Neural Networks," arXiv:2408.06681 (2025).
- M. S. S. Rahman et al., "Universal linear intensity transformations using spatially-incoherent diffractive processors," *Light: Sci. Appl.* **12**, 195 (2023).
