# QWZ‑Gap‑Topology

A combined theoretical and computational study of the Qi–Wu–Zhang (QWZ) Chern‑insulator model, focusing on rigorous proofs of spectral gap opening and quantized Chern‑number jumps, alongside numerical verification and extensions.

## Project Overview

- **Model**: 2D Bloch Hamiltonian  
  $H(\mathbf{k};m) = \sin k_x * \sigma_x + \sin k_y * \sigma_y + \bigl(m + \cos k_x + \cos k_y\bigr) * \sigma_z, \\ \mathbf{k}\in[-\pi,\pi]^2, \\ m\in\mathbb{R}.$
- **Theoretical Goals**  
  1. Prove a lower bound on the spectral gap $\(\Delta(m)\)$ using Kato’s perturbation theory and Weyl’s theorem.  
  2. Rigorously show that the Chern number of the lower band jumps by ±1 at critical values $\(m=-2,0,2\)$.  
- **Computational Goals**  
  - Numerically compute $\(\Delta_0(m)\)$ on a fine $\((k_x,k_y)\)$ grid and compare against analytic bounds.  
  - Implement the Fukui–Hatsugai–Suzuki algorithm to compute Chern numbers and visualize their stepwise changes.  
  - Prototype extensions: disorder‐stability simulations, non‑Hermitian generalizations, and edge‐state calculations.
- **Long‑Term Applications**  
  - Complement with DFT/MD studies of real 2D magnets (e.g., CrI₃) in future work.  
  - Develop publishable extensions in disorder and non‑Hermitian topology.

---
