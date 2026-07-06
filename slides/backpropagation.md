# Training in SVETlANNa: backprop through complex optical fields

A setup is a **differentiable PyTorch module**: the forward pass sends a complex `Wavefront`
through the elements (`FreeSpace` = FFT·transfer·IFFT, `DiffractiveLayer` = `·exp(iφ)`, …).
Training is standard autograd:

```
field → setup(field) → I = |E_out|²  →  loss(I, target)  →  loss.backward()  →  optimizer.step()
```

The trainable parameters (phase masks φ, distances) are real; **the fields are complex**, so
autograd uses **Wirtinger (CR) calculus**.

### Wirtinger calculus — complex differentiation for backprop
Treat `z` and its conjugate `z̄` as **independent** variables:
$$\frac{\partial}{\partial z}=\tfrac12\!\left(\frac{\partial}{\partial x}-i\frac{\partial}{\partial y}\right),\qquad
\frac{\partial}{\partial \bar z}=\tfrac12\!\left(\frac{\partial}{\partial x}+i\frac{\partial}{\partial y}\right).$$

- Optics is **non-holomorphic**: detection `I = |E|² = E·E*` depends on both `E` and `E*`,
  with $\dfrac{\partial I}{\partial \bar E}=E$. The Cauchy–Riemann shortcut ($\partial/\partial\bar z=0$) does **not** apply, so the full Wirtinger chain rule is required.
- For a **real-valued loss** `L`, PyTorch stores the **conjugate Wirtinger derivative**
  $\texttt{z.grad}=\partial L/\partial\bar z$, which is exactly the **steepest-descent direction**, so
  `z ← z − η·z.grad` minimizes `L` with no extra factors.

**Tiny example.** Mask `m`, input field `z`, `E = m·z`, loss `L = |E|²`. The needed gradient is

$$\frac{\partial L}{\partial \bar z}=\underbrace{\frac{\partial L}{\partial E}\frac{\partial E}{\partial \bar z}}_{\bar E\,\cdot\,0}+\underbrace{\frac{\partial L}{\partial \bar E}\frac{\partial \bar E}{\partial \bar z}}_{E\,\cdot\,\bar m}=|m|^2 z .$$

The correct result comes **entirely from the conjugate branch**; the holomorphic-only shortcut
would give `|m|²·z̄` (wrong, conjugated). That conjugate branch is exactly the adjoint `m → m̄`.

### What backprop does physically — the adjoint
Each element is a linear operator `U` on the field; its **vector–Jacobian product is the
adjoint** $U^\dagger$ (conjugate transpose):

| forward | backward (adjoint) |
|---|---|
| free space `FFT·H·IFFT` | `FFT·H*·IFFT` (propagate the **reverse** distance) |
| phase mask `·exp(iφ)` | `·exp(−iφ)` |

This is exactly the elements' `reverse()` methods (`conj(transmission)`): **backpropagation =
running the light backwards through the system**. The real parameter gradients
(e.g. $\partial L/\partial\varphi$) then follow from one more Wirtinger chain-rule step at each mask.

> Forward = light through the optics. Backward = adjoint (conjugate) propagation, via Wirtinger calculus.
