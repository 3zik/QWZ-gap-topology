import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

def qwz_hamiltonian(kx, ky, m):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.sin(kx) * sx + np.sin(ky) * sy + (m + np.cos(kx) + np.cos(ky)) * sz

def lower_band_eigenvector(kx, ky, m):
    H = qwz_hamiltonian(kx, ky, m)
    vals, vecs = eig(H)
    idx = np.argmin(vals.real)
    return vecs[:, idx]

def compute_chern_number(m, N=20):
    """Compute Chern number of lower band via Fukui-Hatsugai-Suzuki on an N×N grid."""
    dk = 2 * np.pi / N
    kx_vals = np.linspace(-np.pi, np.pi - dk, N)
    ky_vals = np.linspace(-np.pi, np.pi - dk, N)
    
    U = np.zeros((N, N, 2), dtype=complex)
    for ix, kx in enumerate(kx_vals):
        for iy, ky in enumerate(ky_vals):
            U[ix, iy, :] = lower_band_eigenvector(kx, ky, m)
    
    def link(u1, u2):
        val = np.vdot(u1, u2)
        return val / np.abs(val)
    
    F = 0.0
    for ix in range(N):
        for iy in range(N):
            ix1 = (ix + 1) % N
            iy1 = (iy + 1) % N
            U00 = U[ix, iy]
            U10 = U[ix1, iy]
            U11 = U[ix1, iy1]
            U01 = U[ix, iy1]
            # plaquette product
            prod = link(U00, U10) * link(U10, U11) * link(U11, U01) * link(U01, U00)
            F += np.angle(prod)
    chern = F / (2 * np.pi)
    return np.round(chern).real

# Compute and plot C(m)
m_values = np.linspace(-3, 3, 61)
chern_values = [compute_chern_number(m, N=20) for m in m_values]

plt.figure(figsize=(6, 4))
plt.step(m_values, chern_values, where='mid', label='Chern number $C(m)$')
plt.xlabel('$m$')
plt.ylabel('$C(m)$')
plt.title('Step Function of the Chern Number vs. Mass Parameter $m$')
plt.grid(True)
plt.tight_layout()
plt.show()
