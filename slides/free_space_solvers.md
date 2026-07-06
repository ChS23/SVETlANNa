# Free-space propagation in SVETlANNa

The `FreeSpace` element propagates a `Wavefront` between planes.
Four solvers, all:

- derived from the **Rayleigh–Sommerfeld** diffraction integral / angular spectrum
- **FFT-based**, fully **differentiable** (autograd), GPU-ready
- support batched & polychromatic fields (per-wavelength `k`)

| method | domain | padding | best for |
|--------|--------|---------|----------|
| `ASM`   | angular spectrum | none | short distance |
| `zpASM` | angular spectrum | zero-padded | short–medium distance |
| `RSC`   | R–S convolution  | none | long distance |
| `zpRSC` | R–S convolution  | zero-padded | long distance |

---

## ASM — Angular Spectrum Method

Multiply the field's spectrum by the propagation transfer function:

$$E(z{+}d)=\mathcal F^{-1}\{\mathcal F\{E\}\cdot H\},\quad
H=e^{\,i\,d\,k_z},\ \ k_z=\sqrt{k^2-k_x^2-k_y^2}$$

- **Pros:** fastest (one FFT pair), exact in the near field.
- **Cons:** circular convolution → **wrap-around / aliasing** if the distance is too large.
- **Validity:** sampling condition $k_{x,\max}\ge k/\sqrt{1+(2d/L_x)^2}$; warns when violated.
- **Use when:** short propagation distances, speed matters.

---

## zpASM — Zero-Padded ASM

Same transfer function, but the field is **zero-padded** (default ½·N each side ⇒ grid doubled) before the FFT to suppress wrap-around.

- **Pros:** removes the circular-convolution artifact of ASM; larger valid range.
- **Cons:** ~4× memory/compute (padded grid).
- **Validity:** below a **critical distance**
  $$d_c=(N+P)\,\sqrt{1-(\lambda/2\Delta x)^2}\,\frac{\Delta x^2}{\lambda}$$
  warns (high-frequency noise) when $d>d_c$ ⇒ switch to RSC.
- **Use when:** short–medium distances where ASM aliases.

---

## RSC — Rayleigh–Sommerfeld Convolution

Convolve with the real-space R–S impulse response (a spherical wave), via FFT:

$$h(x,y;z)=\Big(-ik-\tfrac1r\Big)\frac{z}{r}\,\frac{e^{\,ikr}}{2\pi r},\quad
r=\sqrt{x^2+y^2+z^2}$$

- **Pros:** accurate for **large** distances where the angular-spectrum sampling fails.
- **Cons:** convolution kernel → wrap-around without padding.
- **Validity:** needs distance **above** the critical distance; warns when too short ⇒ prefer (zp)ASM.
- **Use when:** far-field / long propagation.

---

## zpRSC — Zero-Padded RSC

RSC with **zero padding** to remove convolution wrap-around.

- **Pros:** most robust at long distances; cleanest far-field result.
- **Cons:** highest cost (padded convolution).
- **Validity:** same long-distance regime as RSC; warns when the distance is too short.
- **Use when:** long distances and you want artifact-free results.

---

## Choosing a solver

```
        short distance  ───────────────▶  long distance
   ASM ──── zpASM ──── (critical distance d_c) ──── RSC ──── zpRSC
   fast                                                    most robust
```

- Start near field → **ASM**; if it warns about aliasing → **zpASM**.
- Cross the critical distance $d_c$ → **RSC**; for artifact-free far field → **zpRSC**.
- Zero-padded variants trade ~4× cost for the removal of FFT wrap-around.
- The element **emits a warning** whenever the chosen method is used outside its valid range — let it guide the choice.
