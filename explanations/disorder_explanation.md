# Disorder Stability in the QWZ Model

This document provides a concise overview of the Monte Carlo disorder‑stability study implemented in `disorder_stability.py`.

---

## 1. Disorder Model

* **Perturbation**: Random onsite mass term added to each crystal momentum point:
  $V(\mathbf{k}) = \alpha(\mathbf{k})\,\sigma_z,\quad \alpha(\mathbf{k})\sim\mathrm{Uniform}[-W,W]$.
* **Strength**: $W$ is the disorder amplitude.

---

## 2. Numerical Procedure

1. **Clean Gap**: Compute $\Delta_0(m)$ by scanning a k‑grid and finding the minimum band separation of the clean QWZ Hamiltonian.
2. **Monte Carlo Loop**:

   * For each disorder strength $W$:

     1. Generate $N_s$ random disorder realizations of $\alpha(\mathbf{k})$.
     2. At each $\mathbf{k}$, form the perturbed Hamiltonian $H+\alpha(\mathbf{k})\sigma_z$ and compute its two eigenvalues.
     3. Record the minimum gap over the entire Brillouin zone for each realization.
3. **Statistics**: Compute the mean and standard deviation of the perturbed gap across realizations.

---

## 3. Analytic Comparison

* **Weyl Bound**: $\Delta_{\rm pert}(m) \ge \Delta_0(m) - 2W\.$
* **Plot**: Mean perturbed gap vs. $W$ with error bars, overlaid with the line $\Delta_0 - 2W$.

  * Verifies numerically that the gap remains open up to $W\approx\Delta_0/2$.

---

## 4. Physical Implications

* **Robustness criterion**: The topological insulator remains gapped if disorder strength is below half the clean gap.
* **Design guidance**: Quantitative threshold for tolerable disorder in experimental realizations.

---

> **Note:** The plateau of ⟨Δₚₑᵣₜ⟩ for W > ½ Δ₀ reflects that the analytic bound Δ₀ − 2W becomes zero (or negative) in that regime; the system’s minimal gap cannot go below zero, so the mean remains at its “floor.”
