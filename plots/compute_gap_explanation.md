# compute\_gap.py — Purpose & Physical Meaning

This script numerically evaluates the bulk band gap of the Qi–Wu–Zhang Chern-insulator model as a function of the mass parameter \$m\$. It helps illustrate when the system is insulating and identifies topological phase transitions.

---

## 1. Core Functionality

1. **Sweep over the mass parameter \$m\$**

   * Typically in a range (e.g. \[-3.5, 3.5]).
2. **Discretize the Brillouin zone**

   * Create a regular NxN grid of \$k\_x, k\_y\$ in \[-π, π].
3. **Compute band energies**

   * For each (kx, ky) and each \$m\$, call `band_energies(kx, ky, m)` from `qwz_model.py`.
   * Obtain two energies E\_-(k), E\_+(k).
4. **Determine the minimum gap**

   * Compute E\_+(k) - E\_-(k) at each k.
   * Find the minimum over all k:

     ```
     Δ0(m) = min{ E_+(k) - E_-(k) }
     ```
5. **Save & plot**

   * Save (m, Δ0(m)) pairs to `results/gap_vs_m.csv`.
   * Plot Δ0(m) vs. m and save to `plots/gap_vs_m.png`.

---

## 2. Physical Interpretation

* **Band gap Δ0(m):** The energy difference between the conduction and valence bands at half-filling. A nonzero gap indicates an insulating phase.
* **Gap closings:** Values of m where Δ0(m)=0 correspond to topological phase transitions (critical points). For the QWZ model, these occur at m ≈ -2, 0, 2.
* **Insulating vs. metallic behavior:**

  * Δ0(m)>0 ⇒ insulator (bulk states separated by an energy gap).
  * Δ0(m)=0 ⇒ gap closes; system can conduct.
* **Topology:** Changes in m that close and reopen the gap can change the Chern number of the bands, indicating distinct topological phases.

---

## 3. Usage Notes

* Ensure the `results/` and `plots/` directories exist (or create them via `os.makedirs` in the script).
* Adjust the grid resolution (`Nk`) to trade off accuracy vs. computation time.
* Overlay critical lines at m=-2, 0, 2 in the plot for clarity.

---

> This file is a quick reference to remind you of the goals, methods, and physical insights provided by `compute_gap.py`.
