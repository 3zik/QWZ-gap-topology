import numpy as np
from numpy.linalg import eigvalsh

def H_qwz(kx, ky, m):
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    H = np.sin(kx)*sx + np.sin(ky)*sy + (m + np.cos(kx)+np.cos(ky))*sz
    return H

def band_energies(kx, ky, m):
    H = H_qwz(kx, ky, m)
    return eigvalsh(H)
