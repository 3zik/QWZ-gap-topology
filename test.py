import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import itertools

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def hamiltonian(kx, ky, m=-1, γ=0):
    """Non-Hermitian two-band QWZ Hamiltonian"""
    return np.sin(kx) * sigma_x + np.sin(ky) * sigma_y + (m + np.cos(kx) + np.cos(ky) + 1j * γ) * sigma_z

def eigenvalues_and_biorthogonal_projectors(kx, ky, m=-1, γ=0, ε=1e-2):
    H = hamiltonian(kx, ky, m, γ)
    eigvals, righ_vecs = eig(H)
    left_vecs = eig(H.T.conj())[1]  # Left eigenvectors
    projectors = []

    for righ_vec, left_vec, ev in zip(righ_vecs.T, left_vecs.T, eigvals):
        if np.iscomplexobj(ev) and ev.imag != 0:
            continue

        norm = np.dot(left_vec.conj().T, righ_vec)
        projector = np.outer(righ_vec, left_vec.conj().T) / norm

        if np.abs(np.sum((projector @ projector) - projector).real) > ε:
            continue

        projectors.append((projector, np.real(ev)))

    if not projectors:
        return None

    projectors = sorted(projectors, key=lambda x: x[1])
    return projectors[0][0]

def chern_number(gamma_sweeps, N=80, dz=0.01, ε=1e-2):
    bz_points = [(np.pi * i / N, np.pi * j / N) for i in range(-N, N+1, 2) for j in range(-N, N+1, 2)]
    chern_results = []
    plaquette_count_log = []
    stable_count = 0

    for γ in gamma_sweeps:
        plaquette_counts = 0
        stable_counts = 0
        plaquette_sums = []

        for (kx_min, ky_min) in bz_points:
            kx_max = kx_min + dz
            ky_max = ky_min + dz
            P1 = eigenvalues_and_biorthogonal_projectors(kx_min, ky_min, γ=γ, ε=ε)
            P2 = eigenvalues_and_biorthogonal_projectors(kx_max, ky_min, γ=γ, ε=ε)
            P3 = eigenvalues_and_biorthogonal_projectors(kx_max, ky_max, γ=γ, ε=ε)
            P4 = eigenvalues_and_biorthogonal_projectors(kx_min, ky_max, γ=γ, ε=ε)
            
            if (P1 is None) or (P2 is None) or (P3 is None) or (P4 is None):
                continue

            trace_prod = np.trace(P1 @ P2 @ P3 @ P4)
            plaquette_sums.append(trace_prod)
            plaquette_counts += 1
            
            if np.allclose((P1 @ P1), P1, atol=ε):
                stable_counts += 1

        if plaquette_sums:
            arg_sum = np.sum(np.log(plaquette_sums).imag)
            chern = arg_sum / (2 * np.pi)
            chern_results.append((γ, chern))
            stable_count += stable_counts
            plaquette_count_log.append((γ, plaquette_counts))

    print(f"Total stable projectors: {stable_count}")
    for γ_val, plaquettes in plaquette_count_log:
        print(f"γ={γ_val:.2f}: {plaquettes} valid plaquettes")
    return chern_results

def band_data(gamma, kx_range, ky=0, m=-1, ε=1e-2):
    real_energies = []
    imag_energies = []
    valid_flag = True
    
    for kx in kx_range:
        H = hamiltonian(kx, ky, m, gamma)
        eigvals, _ = eig(H)
        
        sorted_real = np.sort(np.real(eigvals))
        sorted_imag = np.sort(np.imag(eigvals))
        
        real_energies.append(sorted_real[0])
        imag_energies.append(sorted_imag[0])
        
        if eigenvalues_and_biorthogonal_projectors(kx, ky, γ=gamma, ε=ε) is None:
            valid_flag = False

    if not valid_flag:
        print(f"Warning: Unstable eigendata regions detected at γ={gamma}")

    return np.array(real_energies), np.array(imag_energies)

def spectrum_data(gamma_values, kx_sample, ky_sample, m=-1):
    all_spectra = []
    for γ in gamma_values:
        eigvals = []
        for ky in ky_sample:
            for kx in kx_sample:
                H = hamiltonian(kx, ky, m, γ)
                vals, _ = eig(H)
                eigvals.extend(vals)
        all_spectra.append((γ, np.array(eigvals)))
    return all_spectra

# Entry point
gamma_sweeps = np.linspace(0, 1, 101)
chern_data = chern_number(gamma_sweeps, dz=0.01, N=80)

gamma_range = [res[0] for res in chern_data]
chern_vals = [np.round(res[1]).astype(int) if abs(res[1] - round(res[1])) < 1e-5 else res[1] for res in chern_data]

# Plot Chern number vs Non-Hermiticity
plt.plot(gamma_range, chern_vals, marker='o')
plt.title("Biorthogonal Chern Number vs Non-Hermiticity (γ)")
plt.xlabel("Non-Hermiticity (γ)")
plt.ylabel("Chern Number")
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 2))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

gamma_sample = np.linspace(0, 1, 5)
ky_cut = 0
kx_range = np.linspace(-np.pi, np.pi, 200)

plt.figure(figsize=(12, 6))
for γ in gamma_sample:    
    real_band, imag_band = band_data(γ, kx_range, ky=ky_cut)
    plt.subplot(1, 2, 1)
    plt.plot(kx_range, real_band, label=f"γ={γ:.2f}")
    plt.xlabel("$k_x$")
    plt.ylabel("Re[E($k_x$)]")
    plt.title("Real part of band structure")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(kx_range, imag_band, label=f"γ={γ:.2f}")
    plt.xlabel("$k_x$")
    plt.ylabel("Im[E($k_x$)]")
    plt.title("Imaginary part of band structure")
    plt.legend()

plt.tight_layout()
plt.show()

kx_sample = np.linspace(-np.pi, np.pi, 15)
ky_sample = np.linspace(-np.pi, np.pi, 15)

for γ, eiglist in spectrum_data(np.linspace(0, 1, 4), kx_sample, ky_sample):
    plt.figure(figsize=(4, 4))
    plt.scatter(np.real(eiglist), np.imag(eiglist), s=1, label="Energies")
    circle1 = plt.Circle((0, 0), γ/2, fill=False, linestyle='--', color='grey')  # Approximation
    plt.gca().add_artist(circle1)
    plt.legend()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("Re[ε]")
    plt.ylabel("Im[ε]")
    plt.title("Point-gap spectra")
    plt.axis('equal')
    plt.show()
