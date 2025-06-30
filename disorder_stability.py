import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

def H_qwz(kx, ky, m):
    """Clean QWZ Hamiltonian at (kx, ky)."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.sin(kx)*sx + np.sin(ky)*sy + (m + np.cos(kx) + np.cos(ky))*sz

def compute_clean_gap(m, Nk=50):
    """Compute clean-limit gap Δ0(m) on an Nk×Nk k-grid."""
    k_vals = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    min_gap = np.inf
    for kx in k_vals:
        for ky in k_vals:
            ev = eigvalsh(H_qwz(kx, ky, m))
            gap = ev[1] - ev[0]
            min_gap = min(min_gap, gap)
    return min_gap

def compute_disordered_gap(m, W, Nk=50, num_samples=100):
    """
    Monte Carlo estimate of perturbed gap for disorder strength W.
    Disorder is modeled as random onsite sigma_z term at each (kx,ky).
    """
    k_vals = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    gaps = []
    for _ in range(num_samples):
        # generate random V at each k
        mags = np.random.uniform(-W, W, size=(Nk, Nk))
        min_gap = np.inf
        for ix, kx in enumerate(k_vals):
            for iy, ky in enumerate(k_vals):
                V = mags[ix, iy] * np.array([[1, 0], [0, -1]], dtype=complex)
                Ht = H_qwz(kx, ky, m) + V
                ev = eigvalsh(Ht)
                gap = ev[1] - ev[0]
                min_gap = min(min_gap, gap)
        gaps.append(min_gap)
    return np.array(gaps)

# Parameters
m = 0.5  # example mass in [0,2]
Nk = 50
num_samples = 200
W_values = np.linspace(0, 1.5, 15)  # disorder strengths

# Compute clean gap
delta0 = compute_clean_gap(m, Nk)

# Monte Carlo sweep
mean_gaps = []
std_gaps = []
for W in W_values:
    sample_gaps = compute_disordered_gap(m, W, Nk, num_samples)
    mean_gaps.append(sample_gaps.mean())
    std_gaps.append(sample_gaps.std())

mean_gaps = np.array(mean_gaps)
std_gaps = np.array(std_gaps)

# Analytic bound line
bound_line = delta0 - 2 * W_values

# Plot results
plt.figure(figsize=(6, 4))
plt.errorbar(W_values, mean_gaps, yerr=std_gaps, fmt='o', label='Monte Carlo mean gap')
plt.plot(W_values, bound_line, '-', label=r'Analytic bound $\Delta_0 - 2W$')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Disorder strength $W$')
plt.ylabel('Mean perturbed gap')
plt.title(f'Disorder Stability at m={m:.1f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
