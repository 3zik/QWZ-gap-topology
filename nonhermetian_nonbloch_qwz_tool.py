# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from numpy.linalg import eig, eigh # eigh for Hermitian, eig for general

# --- Physics Model Definition ---
def H_nonbloch(theta, ky, m, gamma, r):
    """
    Non-Bloch QWZ Hamiltonian.
    theta: kx (real part of non-Bloch momentum)
    ky: ky momentum
    m: mass parameter
    gamma: non-Hermitian gain/loss parameter
    r: GBZ radius for kx direction
    """
    t = 1.0
    kx = theta # theta is kx in the non-Bloch context
    # Note: For the QWZ model, the non-Bloch deformation is typically applied
    # along one direction (e.g., kx). So, ry might not be directly 'r'
    # if ky is kept Bloch-like. Assuming r applies to kx here.
    # If both kx and ky are non-Bloch, this would need to be generalized.

    # Terms for QWZ Hamiltonian (simplified for 2x2)
    # dx, dy, dz are defined with respect to the Bloch momentum.
    # For non-Bloch, kx -> kx + i log(r)
    # cos(kx + i log(r)) = cos(kx)cosh(log(r)) - i sin(kx)sinh(log(r)) = cos(kx)*r_real - i sin(kx)*r_imag
    # sin(kx + i log(r)) = sin(kx)cosh(log(r)) + i cos(kx)sinh(log(r)) = sin(kx)*r_real + i cos(kx)*r_imag

    # For the original QWZ terms:
    # dx = t * (cos(kx) + 1)
    # dy = t * sin(ky)
    # dz = m - t * (cos(kx) + cos(ky))

    # Adapt dx and dz for non-Bloch kx -> kx + i log(r)
    # Your current H_nonbloch(theta, ky, m, gamma, r) uses:
    # dx = t * (rx * cos(kx) + 1) where rx = r
    # This implies that the 't' in the H definition multiplies 'r * cos(kx)' and 'r * sin(ky)'.
    # This is a specific non-Bloch form. Let's stick to your definition:
    dx = t * (r * np.cos(kx) + 1)
    dy = t * (np.sin(ky)) # Assuming ky is still Bloch (not deformed by r)
    dz = m - t * (np.cos(kx) + np.cos(ky)) # This also needs to be adjusted if kx is non-Bloch.

    # Re-evaluating based on standard non-Bloch substitution:
    # If kx -> kx + i log(r), then
    # cos(kx) -> (r*exp(i kx) + (1/r)*exp(-i kx))/2
    # sin(kx) -> (r*exp(i kx) - (1/r)*exp(-i kx))/(2i)
    #
    # Given your current dx and dz, it seems like your `H_nonbloch` has already
    # incorporated some form of non-Bloch structure where `r` directly multiplies
    # `cos(kx)` and `sin(ky)`. Let's clarify if `r` also applies to `cos(ky)` etc.
    # Based on your H_nonbloch:
    # dx = t * (rx * cos(kx) + 1) -> t * (r * cos(theta) + 1)
    # dy = t * (ry * sin(ky))     -> t * (r * sin(ky)) 
    # dz = m - t * (cos(kx) + cos(ky)) -> m - t * (cos(theta) + cos(ky))
    # This suggests that only the `t` term in dx and dy is modified by `r`.
    # Let's assume your definition of H_nonbloch is correct for your model.

    H = np.array([[dz + 1j * gamma, dx - 1j * dy],
                  [dx + 1j * dy, -dz - 1j * gamma]])
    return H

def biorthonormalize(left, right):
    """
    Biorthonormalize a pair of left and right eigenvectors.
    Ensures <left|right> = 1.
    """
    norm = np.dot(np.conj(left), right)
    # Add a small epsilon to avoid division by zero if norm is very close to zero
    return left / np.sqrt(norm + 1e-18), right / np.sqrt(norm + 1e-18)

# --- GBZ Radius Computation ---
def compute_gbz_radius(gamma, m):
    """
    Compute GBZ radius 'r' by minimizing the difference in ABSOLUTE VALUES of eigenvalues
    at a specific kx (e.g., kx=pi), for the non-Bloch QWZ Hamiltonian.
    """
    def eigenvalue_magnitude_spread(logr):
        r_val = np.exp(logr)
        # Use a generic kx point, often pi is a good choice for stability.
        # The GBZ radius typically doesn't depend on ky for the QWZ model
        # when considering the kx deformation.
        H = H_nonbloch(np.pi, 0.0, m, gamma, r_val) # Use kx=pi
        evals = np.linalg.eig(H)[0]

        # The GBZ condition is that the magnitudes of the two eigenvalues are equal.
        # So, we want to minimize the difference between their absolute values.
        return np.abs(np.abs(evals[0]) - np.abs(evals[1]))

    # Optimize for logr in a reasonable range.
    res = minimize_scalar(eigenvalue_magnitude_spread, bounds=(-5, 5), method='bounded')
    return np.exp(res.x)

# --- Non-Bloch Chern Number Computation ---
def compute_nonbloch_chern(Nk, gamma, m):
    """
    Compute non-Bloch Chern number for the QWZ model.
    Nk: Number of k-points in each direction (grid size Nk x Nk)
    gamma: Non-Hermitian parameter
    m: Mass parameter
    """
    # 1. Compute GBZ radius for the given gamma and m
    r = compute_gbz_radius(gamma, m)
    print(f"  > Computed GBZ radius r = {r:.4f} for gamma = {gamma:.4f}")

    thetas = np.linspace(0, 2 * np.pi, Nk, endpoint=False) # kx grid
    kys = np.linspace(0, 2 * np.pi, Nk, endpoint=False)     # ky grid

    # Store left and right eigenvectors for the occupied band
    left_vecs = np.zeros((Nk, Nk, 2), dtype=complex)
    right_vecs = np.zeros((Nk, Nk, 2), dtype=complex)

    # Loop through momentum space to get eigenvectors
    for i, ky in enumerate(kys):
        for j, theta in enumerate(thetas):
            H = H_nonbloch(theta, ky, m, gamma, r)
            
            # Right eigenvectors (from H)
            evalsR, evecsR = eig(H)
            
            # Left eigenvectors (from H.adjoint())
            # For H_L |psi_L> = lambda_L |psi_L>, H_L = H.conj().T
            evalsL, evecsL = eig(H.conj().T)
            
            # Sort eigenvalues by their real part to identify the "occupied" band
            idxR = np.argsort(evalsR.real)
            idxL = np.argsort(evalsL.real)
            
            # Select the eigenvector corresponding to the lowest real part eigenvalue (occupied band)
            vecR_occupied = evecsR[:, idxR[0]]
            vecL_occupied = evecsL[:, idxL[0]]
            
            # Biorthonormalize
            vecL_biortho, vecR_biortho = biorthonormalize(vecL_occupied, vecR_occupied)
            
            right_vecs[i, j, :] = vecR_biortho
            left_vecs[i, j, :] = vecL_biortho

    # Calculate Berry curvature using the Fujiwara-Hatsugai plaquette formula
    berry_curv = np.zeros((Nk, Nk), dtype=float)
    
    for i in range(Nk):
        for j in range(Nk):
            ip = (i + 1) % Nk # k_y + delta k_y
            jp = (j + 1) % Nk # k_x + delta k_x

            # Wavefunctions at the four corners of the plaquette
            # psi(kx, ky) -> psiR00, psiL00
            # psi(kx+dkx, ky) -> psiR10, psiL10
            # psi(kx+dkx, ky+dky) -> psiR11, psiL11
            # psi(kx, ky+dky) -> psiR01, psiL01
            
            psiR00 = right_vecs[i, j]
            psiR10 = right_vecs[i, jp] # Careful with indexing: (ky_idx, kx_idx)
            psiR11 = right_vecs[ip, jp]
            psiR01 = right_vecs[ip, j]

            psiL00 = left_vecs[i, j]
            psiL10 = left_vecs[i, jp]
            psiL11 = left_vecs[ip, jp]
            psiL01 = left_vecs[ip, j]

            # Calculate the parallel transport factors (link variables)
            # U_x(kx, ky) = <L(kx,ky)|R(kx+dkx,ky)>
            u_x_link_00_10 = np.dot(np.conj(psiL00), psiR10)
            # U_y(kx+dkx, ky) = <L(kx+dkx,ky)|R(kx+dkx,ky+dky)>
            u_y_link_10_11 = np.dot(np.conj(psiL10), psiR11)
            # U_x(kx, ky+dky) = <L(kx,ky+dky)|R(kx+dkx,ky+dky)> conjugate for loop direction
            # For the product: U_x(k_x, k_y) U_y(k_x+dk_x, k_y) U_x^{-1}(k_x, k_y+dk_y) U_y^{-1}(k_x, k_y)
            # This means u3 and u4 should be conjugate of the path from 11->01 and 01->00
            # Or, more simply, use the closed loop product formula:
            # P = U_x(i,j) U_y(i,j+1) U_x(i+1,j+1)^-1 U_y(i+1,j)^-1
            # Which is U_x(i,j) U_y(i,j+1) conj(U_x(i+1,j+1)) conj(U_y(i+1,j))

            # The standard gauge invariant plaquette product:
            # P(i,j) = U_x(i,j) * U_y(i,j+1) * U_x(i+1,j)^-1 * U_y(i,j)^-1
            # where U_x(i,j) = <Psi(i,j)|Psi(i,j+1)> (along kx)
            # and U_y(i,j) = <Psi(i,j)|Psi(i+1,j)> (along ky)
            # For non-Hermitian, this is <L_old|R_new>

            # Corrected product for Chern number using biorthogonal states
            # U_kx(ky, kx) = <L(ky,kx)|R(ky,kx+dkx)>
            U_kx_start = np.dot(np.conj(psiL00), psiR10)
            # U_ky(ky, kx+dkx) = <L(ky,kx+dkx)|R(ky+dky,kx+dkx)>
            U_ky_mid_x = np.dot(np.conj(psiL10), psiR11)
            # U_kx(ky+dky, kx) = <L(ky+dky,kx)|R(ky+dky,kx+dkx)> (for the backward step in kx)
            U_kx_end = np.dot(np.conj(psiL01), psiR11) # This needs to be conj() in the product
            # U_ky(ky, kx) = <L(ky,kx)|R(ky+dky,kx)> (for the backward step in ky)
            U_ky_start = np.dot(np.conj(psiL00), psiR01) # This needs to be conj() in the product

            # The product: U_kx(i,j) * U_ky(i,j+1) * U_kx(i+1,j)^(-1) * U_ky(i,j)^(-1)
            # This is equivalent to U_kx(i,j) * U_ky(i,j+1) * conj(U_kx(i+1,j)) * conj(U_ky(i,j))
            # Your original `u` variables were correct for the standard formula:
            # u1: <L(i,j)|R(i,j+1)> (kx step)
            # u2: <L(i,j+1)|R(i+1,j+1)> (ky step)
            # u3: <L(i+1,j+1)|R(i+1,j)> (kx step, backward) - This is where the indexing was swapped for R.
            # u4: <L(i+1,j)|R(i,j)> (ky step, backward) - This is also swapped for R.

            # Re-confirming the plaquette product from Shen et al. or similar:
            # U_x(n, m) = <u_n(k_m) | u_n(k_{m+1})> (n for ky, m for kx)
            # U_y(n, m) = <u_n(k_m) | u_{n+1}(k_m)>

            # For biorthogonal: U_x(n, m) = <L_n(k_m) | R_n(k_{m+1})>
            # U_y(n, m) = <L_n(k_m) | R_{n+1}(k_m)>

            # Plaquette product: U_x(i, j) * U_y(i, j+1) * conj(U_x(i+1, j)) * conj(U_y(i, j))
            # Where (i,j) refers to (ky_idx, kx_idx)

            u_kx_forward = np.dot(np.conj(psiL00), psiR10) # <L(ky,kx)|R(ky,kx+dkx)>
            u_ky_forward = np.dot(np.conj(psiL10), psiR11) # <L(ky,kx+dkx)|R(ky+dky,kx+dkx)>
            u_kx_backward = np.dot(np.conj(psiL01), psiR11) # <L(ky+dky,kx)|R(ky+dky,kx+dkx)>
            u_ky_backward = np.dot(np.conj(psiL00), psiR01) # <L(ky,kx)|R(ky+dky,kx)>

            product = u_kx_forward * u_ky_forward * np.conj(u_kx_backward) * np.conj(u_ky_backward)
            
            # Ensure the product is on the unit circle for angle calculation (numerical stability)
            product /= (np.abs(product) + 1e-12)
            berry_curv[i, j] = np.angle(product)

    chern_number = np.sum(berry_curv) / (2 * np.pi)
    return np.real_if_close(chern_number), r

# --- Plotting Functions (Optional but Recommended) ---
def plot_phase_diagram(gammas, cherns, radii):
    """
    Plots the computed Chern number and GBZ radius vs gamma.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(gammas, cherns, 'o-', label='Non-Bloch Chern number', markersize=4)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel(r'$\gamma$', fontsize=12)
    plt.ylabel('Chern number', fontsize=12)
    plt.title('Non-Bloch Chern Number vs $\gamma$', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.subplot(1, 2, 2)
    plt.plot(gammas, radii, 's-', color='orange', label='GBZ radius $r$', markersize=4)
    plt.xlabel(r'$\gamma$', fontsize=12)
    plt.ylabel('GBZ Radius $r$', fontsize=12)
    plt.title('Numerically Computed GBZ Radius vs $\gamma$', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_spectrum(gamma, m, r_val=None, k_points=100):
    """
    Plots the energy spectrum E_n(kx) for a fixed ky (e.g., ky=0) with GBZ deformation.
    Optionally, compute r_val if not provided.
    """
    if r_val is None:
        r_val = compute_gbz_radius(gamma, m)
        print(f"Computed GBZ radius r = {r_val:.4f} for gamma = {gamma:.4f} (for spectrum plot)")

    thetas = np.linspace(-np.pi, np.pi, k_points)
    energies = np.zeros((k_points, 2), dtype=complex)

    for i, theta in enumerate(thetas):
        H = H_nonbloch(theta, 0.0, m, gamma, r_val) # Fix ky=0 for 1D slice
        evals = eig(H)[0]
        energies[i, :] = np.sort(evals.real) + 1j * np.sort(evals.imag) # Sort by real part

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(thetas, energies[:, 0].real, 'b-', label='Band 1 (Re)')
    plt.plot(thetas, energies[:, 1].real, 'r-', label='Band 2 (Re)')
    plt.xlabel(r'$k_x$', fontsize=12)
    plt.ylabel(r'Re(E)', fontsize=12)
    plt.title(f'Real part of Energy Spectrum ($\\gamma={gamma:.2f}$, $r={r_val:.2f}$)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.subplot(1, 2, 2)
    plt.plot(thetas, energies[:, 0].imag, 'b--', label='Band 1 (Im)')
    plt.plot(thetas, energies[:, 1].imag, 'r--', label='Band 2 (Im)')
    plt.xlabel(r'$k_x$', fontsize=12)
    plt.ylabel(r'Im(E)', fontsize=12)
    plt.title(f'Imaginary part of Energy Spectrum ($\\gamma={gamma:.2f}$, $r={r_val:.2f}$)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()

# --- Main Execution Block (for direct script running or notebook cells) ---
if __name__ == '__main__':
    Nk = 300 # Number of k-points for Chern calculation
    m = 1.0 # Mass parameter
    gammas = np.linspace(-2.0, 2.0, 30) # Range of gamma values

    cherns = []
    radii = []

    print("Starting computation of non-Bloch Chern numbers and GBZ radii...")
    for gamma in gammas:
        chern_val, r_val = compute_nonbloch_chern(Nk, gamma, m)
        cherns.append(chern_val)
        radii.append(r_val)
        print(f"  > Gamma = {gamma:.4f}, Chern = {chern_val:.4f}\n")

    print("\nComputation complete. Plotting results...")
    plot_phase_diagram(gammas, cherns, radii)

    # Example: Plot spectrum for a specific gamma value
    plot_spectrum(gamma=1.0, m=1.0)
    plot_spectrum(gamma=0.0, m=1.0) # Hermitian case
    plot_spectrum(gamma=-1.0, m=1.0)
