"""
nonhermitian_gap.py

Compute and plot the point-gap of the non-Hermitian QWZ model:
    H_γ(k; m) = H_0(k; m) + i γ σ_z

The point-gap at zero is δ_pt(m, γ) = min_k min_{λ ∈ σ(H_γ)} |λ|.
We sweep γ ∈ [0, γ_max] and compute δ_pt for each.

Usage:
    python nonhermitian_gap.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from qwz_model import H_qwz  # assumes H_qwz(kx, ky, m) returns the 2×2 Hermitian part

# Parameters
m = 0.5             # mass parameter (choose a gapped m)
Nk = 100            # grid resolution in k-space
gamma_values = np.linspace(0, 1.0, 101)  # sweep of non-Hermitian strength γ
results_dir = "results_nh"
plots_dir = "plots_nh"

# Ensure output directories exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Precompute k-grid
kx_vals = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
ky_vals = np.linspace(-np.pi, np.pi, Nk, endpoint=False)

def compute_point_gap(m, gamma):
    """
    Compute the point-gap δ_pt(m, γ) = min_k min |eigenvalues(H_0 + iγσ_z)|.
    """
    min_dist = np.inf
    for kx in kx_vals:
        for ky in ky_vals:
            H0 = H_qwz(kx, ky, m)
            V = 1j * gamma * np.array([[1, 0], [0, -1]], dtype=complex)
            Ht = H0 + V
            vals = eigvals(Ht)
            dist = np.min(np.abs(vals))
            if dist < min_dist:
                min_dist = dist
    return min_dist

# Sweep gamma and compute δ_pt
delta_pt_values = []
for gamma in gamma_values:
    delta_pt = compute_point_gap(m, gamma)
    delta_pt_values.append(delta_pt)
delta_pt_values = np.array(delta_pt_values)

# Save results to CSV
csv_path = os.path.join(results_dir, f"point_gap_m{m:.2f}.csv")
np.savetxt(csv_path,
           np.column_stack((gamma_values, delta_pt_values)),
           delimiter=",",
           header="gamma,delta_pt",
           comments="")

# Plot δ_pt vs γ
plt.figure(figsize=(6, 4))
plt.plot(gamma_values, delta_pt_values, label=r'$\delta_{\rm pt}(m,\gamma)$')
# Overlay analytic bound Δ0/2 - γ if desired:


from compute_gap import compute_clean_gap
delta0 = compute_clean_gap(m, Nk=Nk)
bound = 0.5 * delta0 - gamma_values
plt.plot(gamma_values, bound, '--', label=r'$\tfrac12\Delta_0 - \gamma$')


plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel(r'Non‑Hermitian strength $\gamma$')
plt.ylabel(r'Point‑gap $\delta_{\rm pt}$')
plt.title(f'Point‑Gap vs. γ for m={m:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plots_dir, f"point_gap_m{m:.2f}.png")
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"Saved CSV to {csv_path}")
print(f"Saved plot to {plot_path}")
