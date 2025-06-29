# QWZ‑Gap‑Topology

A combined theoretical and computational study of the Qi–Wu–Zhang (QWZ) Chern‑insulator model, focusing on rigorous proofs of spectral gap opening and quantized Chern‑number jumps, alongside numerical verification and extensions.

## Project Overview

- **Model**: 2D Bloch Hamiltonian  
  $H(\mathbf{k};m) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y + \bigl(m + \cos k_x + \cos k_y\bigr)\,\sigma_z, \quad \mathbf{k}\in[-\pi,\pi]^2, m\in\mathbb{R}.$
- **Theoretical Goals**  
  1. Prove a lower bound on the spectral gap \(\Delta(m)\) using Kato’s perturbation theory and Weyl’s theorem.  
  2. Rigorously show that the Chern number of the lower band jumps by ±1 at critical values \(m=-2,0,2\).  
- **Computational Goals**  
  - Numerically compute \(\Delta_0(m)\) on a fine \((k_x,k_y)\) grid and compare against analytic bounds.  
  - Implement the Fukui–Hatsugai–Suzuki algorithm to compute Chern numbers and visualize their stepwise changes.  
  - Prototype extensions: disorder‐stability simulations, non‑Hermitian generalizations, and edge‐state calculations.
- **Long‑Term Applications**  
  - Complement with DFT/MD studies of real 2D magnets (e.g., CrI₃) in future work.  
  - Develop publishable extensions in disorder and non‑Hermitian topology.

---

## Timeline (July 2025 – April 2026)

| Month(s)         | Activities & Milestones                                                                                                                                  |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **July – August** | • Deep‑dive into QWZ literature, Kato theory, Berry curvature methods  
  • Scaffold GitHub repo & environment  
  • Initial `qwz_model.py` with band‑energy functions                                                 |
| **September**    | • `compute_gap.py`: scan m vs. Δ₀(m) on 100×100 grid  
  • `chern_calc.py`: implement and test discrete Chern calculation  
  • Draft proof outline for Theorem 1 (gap bound)                                                 |
| **October**      | • Formal write‑up of Theorem 1 in LaTeX  
  • Numerical verification: analytic vs. numeric gap plots (Figure 1)                              |
| **November**     | • Formal write‑up of Theorem 2 (Chern jumps)  
  • Berry curvature heatmaps and local Dirac expansion visualizations                               |
| **December**     | • Prototype Anderson‑disorder stability simulations  
  • Prototype non‑Hermitian QWZ model & point‑gap computations                                      |
| **January 2026** | • Integrate Sections I–VI into cohesive LaTeX draft  
  • Prepare Appendix with code listings                                                          |
| **February**     | • Convergence tests & final high‑res figures  
  • Department seminar or poster                                                                  |
| **March**        | • Full draft of extensions (Section VII)  
  • Incorporate advisor feedback                                                                  |
| **April**        | • Final revisions, references, and formatting  
  • Submit to journal or arXiv; tag code repo v1.0                                                |

---

## Immediate Next Steps

1. **Fork & clone** this repository.  
2. **Create** a Python 3.9 virtual environment and install dependencies (`numpy`, `scipy`, `matplotlib`).  
3. **Begin** by implementing `qwz_model.py` and testing `band_energies(kx, ky, m)`.  
4. **Schedule** a meeting with advisors to review timeline and code stub.

---

> _Let’s get started on making rigorous topology meet hands‑on numerics!_  
