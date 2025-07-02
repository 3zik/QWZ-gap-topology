from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from numpy.linalg import eig

def H_nonbloch(theta, ky, m, gamma, r):
    """Non-Bloch QWZ Hamiltonian."""
    t = 1.0
    kx = theta
    rx = r
    ry = r

    dx = t * (rx * cos(kx) + 1)
    dy = t * (ry * sin(ky))
    dz = m - t * (cos(kx) + cos(ky))
    H = array([[dz + 1j * gamma, dx - 1j * dy],
               [dx + 1j * dy, -dz - 1j * gamma]])
    return H

def biorthonormalize(left, right):
    norm = dot(conj(left), right)
    return left / sqrt(norm), right / sqrt(norm)

def compute_gbz_radius_quartic(gamma, m):
    """Compute GBZ radius using the transfer matrix quartic equation."""
    # Coefficients from transfer matrix approximation
    def characteristic_polynomial_modulus(logr):
        r = exp(logr)
        kx = pi  # use time-reversal symmetric point for stability
        ky = 0.0  # any fixed ky is okay for 1D slice
        Hk = H_nonbloch(kx, ky, m, gamma, r)
        eigvals = eig(Hk)[0]
        return abs(imag(eigvals[0]))  # we want minimal imaginary part

    res = minimize_scalar(characteristic_polynomial_modulus, bounds=(-1, 1), method='bounded')
    return exp(res.x)

def compute_nonbloch_chern(Nk, gamma, m):
    """Compute non-Bloch Chern number given gamma and mass m."""
    r = compute_gbz_radius_quartic(gamma, m)
    thetas = linspace(0, 2 * pi, Nk, endpoint=False)
    kys = linspace(0, 2 * pi, Nk, endpoint=False)
    left_vecs = zeros((Nk, Nk, 2), dtype=complex)
    right_vecs = zeros((Nk, Nk, 2), dtype=complex)

    for i, ky in enumerate(kys):
        for j, theta in enumerate(thetas):
            H = H_nonbloch(theta, ky, m, gamma, r)
            evalsR, evecsR = eig(H)
            evalsL, evecsL = eig(H.conj().T)
            idx = argsort(evalsR.real)
            idxL = argsort(evalsL.real)
            vecR = evecsR[:, idx[0]]
            vecL = evecsL[:, idxL[0]]
            vecL, vecR = biorthonormalize(vecL, vecR)
            right_vecs[i, j, :] = vecR
            left_vecs[i, j, :] = vecL

    berry_curv = zeros((Nk, Nk))
    eps = 1e-12
    for i in range(Nk):
        for j in range(Nk):
            ip = (i + 1) % Nk
            jp = (j + 1) % Nk

            psiR00 = right_vecs[i, j]
            psiR10 = right_vecs[ip, j]
            psiR11 = right_vecs[ip, jp]
            psiR01 = right_vecs[i, jp]

            psiL00 = left_vecs[i, j]
            psiL10 = left_vecs[ip, j]
            psiL11 = left_vecs[ip, jp]
            psiL01 = left_vecs[i, jp]

            u1 = dot(conj(psiL00), psiR10)
            u2 = dot(conj(psiL10), psiR11)
            u3 = dot(conj(psiL11), psiR01)
            u4 = dot(conj(psiL01), psiR00)

            product = u1 * u2 * u3 * u4
            product /= abs(product) + eps
            berry_curv[i, j] = angle(product)

    chern_number = sum(berry_curv) / (2 * pi)
    return real_if_close(chern_number), r

def main():
    Nk = 40
    m = 1.0
    gammas = linspace(-2.0, 2.0, 30)
    cherns = []
    radii = []

    print("Computing non-Bloch Chern numbers with full quartic GBZ radius...")
    for gamma in gammas:
        c, r = compute_nonbloch_chern(Nk, gamma, m)
        cherns.append(c)
        radii.append(r)
        print(f"Computed GBZ radius r = {r:.4f} for gamma = {gamma: .4f}")
        print(f"gamma = {gamma: .4f}, Chern = {c: .4f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gammas, cherns, 'o-', label='Chern number')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Chern number')
    plt.title('Non-Bloch Chern Number vs $\gamma$')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(gammas, radii, 's-', color='orange', label='GBZ radius')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('r (GBZ)')
    plt.title('Numerically Computed GBZ Radius')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

main()
