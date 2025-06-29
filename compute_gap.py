import numpy as np
import matplotlib.pyplot as plt
from qwz_model import band_energies

# Grid settings
Nk = 100  # grid resolution in k-space
m_values = np.linspace(-3.5, 3.5, 100)  # sweep over m

kx_vals = np.linspace(-np.pi, np.pi, Nk)
ky_vals = np.linspace(-np.pi, np.pi, Nk)

gap_values = []

# Main loop: sweep over m
for m in m_values:
    min_gap = np.inf
    for kx in kx_vals:
        for ky in ky_vals:
            energies = band_energies(kx, ky, m)
            gap = energies[1] - energies[0]  # E_+ - E_-
            if gap < min_gap:
                min_gap = gap
    gap_values.append(min_gap)

# Convert to numpy arrays
m_values = np.array(m_values)
gap_values = np.array(gap_values)

# Save to CSV
np.savetxt("results/gap_vs_m.csv", np.column_stack((m_values, gap_values)), delimiter=",", header="m,gap", comments="")

# Plot
plt.figure(figsize=(6, 4))
plt.plot(m_values, gap_values, label="Δ₀(m)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("m")
plt.ylabel("Minimum Band Gap Δ₀(m)")
plt.title("Band Gap vs. Mass Parameter m")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/gap_vs_m.png", dpi=150)
plt.show()
