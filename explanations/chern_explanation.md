# Chern Number Jump in the QWZ Model

This document summarizes the key components of the Chern number jump proof, the numerical implementation in `chern_calc.py`, the resulting step‑function graph of $C(m)$, and the physical meaning and implications of these results.

---

## 1. Rigorous Chern‑Jump Proof

1. **Definition of Chern Number**

   * For the lower band of $H(\mathbf{k};m)$, compute the Berry curvature:

     $\Omega(\mathbf{k}) = i\Bigl(\langle\partial_{k_x}u_-|\partial_{k_y}u_-\rangle - \langle\partial_{k_y}u_-|\partial_{k_x}u_-\rangle\Bigr)$
   * Integrate over the Brillouin zone (BZ):

     $C(m) = \frac{1}{2\pi} \int_{BZ} \Omega(\mathbf{k})\,d^2k \in \mathbb{Z}.$

2. **Gap‑Closing Analysis**

   * Solve $\mathbf{d}(\mathbf{k};m)=0$ for $\sin k_x=0, \sin k_y=0, m+\cos k_x+\cos k_y=0$.
   * Critical points: $(m,\mathbf{k})=(-2,(0,0)), (0,(0,\pi)), (0,(\pi,0)), (2,(\pi,\pi))$.

3. **Piecewise Constancy of $C(m)$**

   * For $m$ in each interval between critical values, the gap remains open and $C(m)$ is constant by continuity.

4. **Local Dirac Expansion**

   * Near $(m_c,\mathbf{k}_c)$, set $m=m_c+\mu$, $\mathbf{k}=\mathbf{k}_c+\mathbf{q}$.
   * Approximate $H\approx q_x\sigma_x + q_y\sigma_y + \mu\sigma_z$, a massive Dirac model.

5. **Dirac Chern Contribution**

   * The 2D Dirac Hamiltonian has Chern $C_D(\mu)=-\tfrac12\mathrm{sgn}(\mu)$.
   * Each gap‑closing Dirac cone contributes $\pm\tfrac12$ when its mass changes sign.

6. **Total Jump**

   * Four Dirac points yield integer jumps of $\pm1$ at $m=-2,0,2$, giving

   $ C(m)=\begin{cases} 0, & m<-2,\\ +1, & -2<m<0,\\ -1, & 0<m<2,\\ 0, & m>2.\end{cases} $
   
---

## 2. Numerical Implementation: `chern_calc.py`

* **Algorithm**: Fukui–Hatsugai–Suzuki method on an $N\times N$ grid.

* **Key Steps**:

  1. Discretize BZ: $k_x,k_y\in[-\pi,\pi)$.
  2. Compute lower‑band eigenvector $|u_-(k_x,k_y)\rangle$.
  3. Form gauge link variables $U_{\mu}(k) = \langle u_-(k)|u_-(k+\delta k_\mu)\rangle/|\cdots|$.
  4. Compute plaquette product $\prod U$ and sum $\mathrm{arg} $ to get total Berry flux.
  5. Normalize by $2\pi$ and round to nearest integer.

* **Sample Code Snippet**:

  ```python
  def compute_chern_number(m, N=20):
      # ... assemble U[ix,iy] = lower_band_eigenvector(kx, ky, m) ...
      F = 0
      for ix in range(N):
          for iy in range(N):
              # indices with periodic BC
              # compute plaquette product of U's
              F += np.angle(prod)
      return np.round(F/(2*np.pi))
  ```

---

## 3. Resulting Graph of $C(m)$

![Step Function of C(m)](chern_vs_m.png)

* **Observed Behavior**: Plateaus at 0, +1, –1, 0 with jumps at $m=-2,0,2$.
* **Grid**: 61 evenly spaced $m$ values in $[-3,3]$, $N=20$ grid in BZ.

---

## 4. Physical Meaning

* **Chern number** equals the quantized Hall conductivity (in units of $e^2/h$).
* **Plateaus** correspond to distinct insulating phases with different edge state counts.
* **Jumps** occur when the bulk gap closes—i.e., a topological phase transition.

---

## 5. Implications

1. **Robustness**: The quantization of $C(m)$ is protected as long as the gap remains open.
2. **Experimental Tuning**: Varying a parameter analogous to $m$ in real materials or cold atoms can drive transitions between topological phases.
3. **Design Guidelines**: The location of critical $m$ values pinpoints where to look for gap closings in band-structure engineering.

---

hahah markdwon hahaha its not confusing
