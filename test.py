import numpy as np
from scipy.linalg import eig, logm
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        # Select using real(E)
        if np.iscomplexobj(ev) and ev.imag != 0:
            continue  # Skip if non-hermitian correction makes it invalid

        norm = np.dot(left_vec.conj().T, righ_vec)
        projector = np.outer(righ_vec, left_vec.conj().T) / norm
        
        # Check projector validity
        if np.abs(np.sum((projector @ projector) - projector).real) > ε:
            continue  # Disqualify unstable projectors

        projectors.append((projector, np.real(ev)))
    
    # Select lowest real eigenvalue
    if not projectors:
        return None  # No valid projectors left

    projectors = sorted(projectors, key=lambda x: x[1])
    return projectors[0][0]  # Return the lowest biorthogonal projector

def chern_number(gamma_sweeps, N=80, dz=0.01, ε=1e-2):
    bz_points = [(np.pi * i / N, np.pi * j / N) for i in range(-N, N+1, 2) for j in range(-N, N+1, 2)]
    chern_results = []
    plaquette_count_log = []
    stable_count = 0

    for γ in tqdm(np.arange(0, 1.01, dz)):
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
            
            if None in [P1, P2, P3, P4]:
                continue  # Skip unstable region

            trace_prod = np.trace(P1 @ P2 @ P3 @ P4)
            plaquette_sums.append(trace_prod)
            plaquette_counts += 1
            
            if np.allclose((P1 @ P1), P1, atol=ε):
                stable_counts += 1

        if plaquette_sums:
            arg_sum = np.sum(np.log(plaquette_sums).imag)  # Wrapped phase handling
            chern = arg_sum / (2 * np.pi)
            chern_results.append((γ, chern))
            stable_count += stable_counts
            plaquette_count_log.append((γ, plaquette_counts))

    print(f"Total stable projectors: {stable_count}")
    for γ_val, plaquettes in plaquette_count_log:
        print(f"γ={γ_val:.2f}: {plaquettes} valid plaquettes")
    return chern_results

# Calculate Chern number and diagnostic plots
chern_data = chern_number(dz=0.01, N=80)
gamma_range = [res[0] for res in chern_data]
chern_vals = [np.round(res[1]).astype(int) if abs(res[1] - round(res[1])) < 1e-5 else res[1] for res in chern_data]

# Plot Chern number vs Non-Hermiticity
plt.plot(gamma_range, chern_vals, marker='o')
plt.title("Biorthogonal Chern Number vs Non-Hermiticity (γ)")
plt.xlabel("Non-Hermiticity (γ)")
plt.ylabel("Chern Number")
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 2))  # Assuming integer jumps within range
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Additional diagnostic visualizations here...
def band_data(gamma, kx_range, ky=0, m=-1, ε=1e-2):
    real_energies = []
    imag_energies = []
    valid_flag = True
    
    for kx in kx_range:
        H = hamiltonian(kx, ky, m, gamma)
        eigvals, _ = eig(H)
        
        # Track real part of lowest eigenvalue
        sorted_real = np.sort(np.real(eigvals))
        sorted_imag = np.sort(np.imag(eigvals))
        
        real_energies.append(sorted_real[0])
        imag_energies.append(sorted_imag[0])
        
        # Check valid projector existence (flag for potential plot labeling later)
        if eigenvalues_and_biorthogonal_projectors(kx, ky, γ=gamma, ε=ε) is None:
            valid_flag = False

    if not valid_flag:
        print(f"Warning: Unstable eigendata regions detected at γ={gamma}")

    return np.array(real_energies), np.array(imag_energies)

# Add point-gap spectra calculations
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

# Trace norm visualization (heat grid)
def trace_norm_grid(gamma, kx_range, ky_range, m=-1, ε=1e-2):
    norms = []
    for kx, ky in tqdm(itertools.product(kx_range, ky_range), desc="Trace Norm Grid"):
        P = eigenvalues_and_biorthogonal_projectors(kx, ky, γ=gamma, m=m, ε=ε)
        if P is None:
            norms.append(np.nan)
        else:
            norms.append(np.trace(P @ P - P))
            
    return np.array(norms).reshape((len(ky_range), len(kx_range)))
    
# Add visualization of band structure
def plot_band_structure(chern_data):
    gamma_sample = np.linspace(0, 1, 5)
    ky_cut = 0
    kx_range = np.linspace(-np.pi, np.pi, 200)
    plt.figure(figsize=(12, 6))

    # Plot real and imag parts
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

# Point-gap spectra plots
def plot_point_gap_spectra(spectrum_data, γ_sample):
    kx_sample = np.linspace(-np.pi, np.pi, 15)
    ky_sample = np.linspace(-np.pi, np.pi, 15)

    for γ, eiglist in spectrum_data:
        plt.figure(figsize=(4, 4))
        plt.scatter(np.real(eiglist), np.imag(eiglist), s=1, label="Energies")
        circle1 = plt.Circle((0, 0), γ/2, fill=False, linestyle='--', color='grey')  # Approximation
        plt.gca().add_artist(circle1)
        plt.legend()
        plt.xlim([-γ*2, γ*2])  # Expand window for non-Hermiticity
        plt.ylim([-γ*2, γ*2])
        plt.xlabel("Re[ε]")
        plt.ylabel("Im[ε]")
        plt.title("Point-gap spectra")
        plt.axis('equal')
        plt.show()
    
# Visualization of trace norm heat grid
def plot_trace_norm_grid(gamma_sample):
    for γ in gamma_sample:
        norm_grid = trace_norm_grid(γ, np.linspace(-np.pi, np.pi, 40), np.linspace(-np.pi, np.pi, 40))
        plt.figure(figsize=(5, 5))
        cax = plt.imshow(norm_grid, extent=(-np.pi, np.pi, -np.pi, np.pi), origin='lower')
        cbar = plt.colorbar(cax)
        plt.title(f"Trace Norm Grid for γ={γ}")
        plt.xlabel("$k_x$")
        plt.ylabel("$k_y$")
        cbar.ax.set_ylabel("$\|P^2 - P\|$ (projector stability)")

gamma_values = np.linspace(0, 1, 21)
spectrum_data = spectrum_data(gamma_values, np.linspace(-np.pi, np.pi, 15), np.linspace(-np.pi, np.pi, 15))
plot_band_structure(chern_data)
plot_point_gap_spectra(spectrum_data, gamma_values)  # Show key γ slices
plot_trace_norm_grid((0.0, 0.5, 1.0))


# Compute Chern number results first (main entry point)
chern_data = chern_number(dz=0.01, N=80)  # Existing code from initial script

# Perform visualization calls with data
plot_band_structure(chern_data)  # Use existing chern_data structure

# Obtain spectrum data and visualize separately
gamma_values = np.linspace(0, 1, 21)
spectrum_data = spectrum_data(gamma_values, np.linspace(-np.pi, np.pi, 15), np.linspace(-np.pi, np.pi, 15))
plot_point_gap_spectra(spectrum_data, gamma_values)  # Requires separate dataset collection to show γ series

# Generate and display all γ heat grid diagnostics
plot_trace_norm_grid((0.0, 0.5, 1.0))
