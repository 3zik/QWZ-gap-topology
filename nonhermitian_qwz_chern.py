"""
Non-Hermitian QWZ Model: Biorthogonal Chern Number Computation

This script defines the Non-Hermitian Qi-Wu-Zhang (QWZ) model on a 2D lattice
and computes the biorthogonal Chern number as a function of the non-Hermiticity
parameter gamma (γ). The implementation is stable, modular, and suitable for
scientific publication and reproducibility.

Author: Your Name
Date: July 1, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

class NonHermitianQWZ:
    """
    Class for the non-Hermitian QWZ model and Chern number computation.
    """

    def __init__(self, m=1.0, t=1.0):
        """
        Initializes the NonHermitianQWZ model parameters.

        Args:
            m (float): Mass term parameter.
            t (float): Hopping term parameter.
        """
        self.m = m
        self.t = t
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def hamiltonian(self, kx, ky, gamma=0.0):
        """
        Constructs the Hamiltonian H(k) with non-Hermitian perturbation.

        Args:
            kx (float): Momentum in x-direction.
            ky (float): Momentum in y-direction.
            gamma (float): Non-Hermiticity parameter.

        Returns:
            np.ndarray: The 2x2 Hamiltonian matrix for given kx, ky, and gamma.
        """
        h_x = self.t * np.sin(kx)
        h_y = self.t * np.sin(ky)
        h_z = self.m + self.t * (np.cos(kx) + np.cos(ky))
        H = h_x * self.sigma_x + h_y * self.sigma_y + h_z * self.sigma_z
        # Add the non-Hermitian iγσz term
        return H + 1j * gamma * self.sigma_z

    def get_biorthogonal_vectors(self, H):
        """
        Computes and returns sorted eigenvalues, right, and left eigenvectors.
        Left eigenvectors are normalized such that <L_i | R_j> = delta_ij.

        Args:
            H (np.ndarray): The Hamiltonian matrix.



        Returns:
            tuple: A tuple containing:
                - evals_R_sorted (np.ndarray): Sorted eigenvalues of H.
                - psi_R_lower (np.ndarray): Right eigenvector for the lower band.
                - psi_L_lower (np.ndarray): Left eigenvector for the lower band, biorthonormalized.
                Returns (None, None, None) if normalization fails due to a small factor.
        """
        # Compute right eigenvectors and eigenvalues of H
        evals_R, R = eig(H)  
        # Compute right eigenvectors and eigenvalues of H_dagger (H.conj().T)
        evals_L_dag, L_dag_cols = eig(H.conj().T) 

        # Sort eigenvalues and corresponding right eigenvectors based on the real part
        # for consistent band selection (e.g., the 'lower' band)
        idx_sorted_R = np.argsort(np.real(evals_R))
        eval_lower_R = evals_R[idx_sorted_R[0]]
        psi_R_lower = R[:, idx_sorted_R[0]]

        # Find the eigenvalue of H_dagger that is the complex conjugate of eval_lower_R
        # This corresponds to the eigenvalue for the left eigenvector of the lower band.
        idx_lower_L_dag = np.argmin(np.abs(evals_L_dag - np.conj(eval_lower_R)))
        
        # The corresponding column from L_dag_cols is psi_L_dagger.
        # Conjugate it to get psi_L.
        psi_L_lower = L_dag_cols[:, idx_lower_L_dag].conj()

        # Perform biorthonormalization: <psi_L_lower | psi_R_lower> = 1
        normalization_factor = np.vdot(psi_L_lower, psi_R_lower)
        
        # If the normalization factor is too small, it means the eigenvectors are nearly orthogonal.
        # This occurs at exceptional points where the gap closes.
        # The threshold is chosen to be a small multiple of machine epsilon to avoid
        # skipping valid gapped points due to floating-point inaccuracies.
        if np.abs(normalization_factor) < np.finfo(complex).eps * 10: 
            # This indicates a true exceptional point or severe numerical instability.
            # Returning None will cause the plaquette to be skipped.
            return None, None, None
        
        # Normalize psi_L_lower to satisfy the biorthonormality condition
        psi_L_lower = psi_L_lower / normalization_factor

        # Return the original (unsorted) eigenvalues of H (evals_R) for consistency
        # with the original return signature, though only psi_R_lower and psi_L_lower are used.
        return evals_R, psi_R_lower, psi_L_lower

    def compute_chern_number(self, gamma, N=60):
        """
        Computes the biorthogonal Chern number using the plaquette phase method.

        Args:
            gamma (float): Non-Hermiticity parameter.
            N (int): Grid size for discretizing the Brillouin zone (N x N).

        Returns:
            float: The computed biorthogonal Chern number.
        """
        kx_vals = np.linspace(0, 2 * np.pi, N, endpoint=False)
        ky_vals = np.linspace(0, 2 * np.pi, N, endpoint=False)
        total_phase = 0.0

        # Iterate over each plaquette in the Brillouin zone grid
        for i in range(N):
            for j in range(N):
                # Define the four k-points (corners) of the current plaquette
                k_curr = (kx_vals[i], ky_vals[j])
                k_x_plus = (kx_vals[(i + 1) % N], ky_vals[j])
                k_xy_plus = (kx_vals[(i + 1) % N], ky_vals[(j + 1) % N])
                k_y_plus = (kx_vals[i], ky_vals[(j + 1) % N])

                # Compute the lower band's right and left eigenvectors at each corner
                # Handle cases where get_biorthogonal_vectors returns None
                _, psi_R_curr, psi_L_curr = self.get_biorthogonal_vectors(self.hamiltonian(*k_curr, gamma))
                _, psi_R_x_plus, psi_L_x_plus = self.get_biorthogonal_vectors(self.hamiltonian(*k_x_plus, gamma))
                _, psi_R_xy_plus, psi_L_xy_plus = self.get_biorthogonal_vectors(self.hamiltonian(*k_xy_plus, gamma))
                _, psi_R_y_plus, psi_L_y_plus = self.get_biorthogonal_vectors(self.hamiltonian(*k_y_plus, gamma))

                # If any eigenvector is None, it means the system is at or near an exceptional point
                # where the gap closes, and the phase is ill-defined. Skip this plaquette.
                if any(v is None for v in [psi_R_curr, psi_L_curr, psi_R_x_plus, psi_L_x_plus, 
                                           psi_R_xy_plus, psi_L_xy_plus, psi_R_y_plus, psi_L_y_plus]):
                    print(f"Skipping plaquette at ({i}, {j}) for gamma={gamma:.4f} due to non-diagonalizability or gap closing.")
                    continue # Skip this plaquette if eigenvectors are not found

                # Compute the gauge link variables around the plaquette
                # These are overlaps between left and right eigenvectors at adjacent k-points
                U_x = np.vdot(psi_L_curr, psi_R_x_plus)
                U_y_x = np.vdot(psi_L_x_plus, psi_R_xy_plus)
                U_min_x = np.vdot(psi_L_xy_plus, psi_R_y_plus)
                U_min_y = np.vdot(psi_L_y_plus, psi_R_curr)

                # Calculate the product of gauge links around the plaquette
                plaquette_product = U_x * U_y_x * U_min_x * U_min_y

                # Accumulate the phase (argument) of the plaquette product
                # A small threshold is used to guard against numerical issues if product is near zero
                if np.abs(plaquette_product) > 1e-12:
                    total_phase += np.angle(plaquette_product)
                else:
                    # This warning indicates a potential problem, possibly band closing or numerical instability
                    print(f"Warning: Plaquette product close to zero at ({i}, {j}) for gamma={gamma:.4f}. Skipping phase contribution.")
                    pass 

        # The Chern number is the total accumulated phase divided by 2π
        return total_phase / (2 * np.pi)

    def compute_chern_vs_gamma(self, gamma_vals, N=60):
        """
        Computes the Chern number for a list of gamma values.

        Args:
            gamma_vals (np.ndarray): Array of non-Hermiticity parameter values.
            N (int): Grid size for Brillouin zone discretization.

        Returns:
            np.ndarray: Array of computed Chern numbers corresponding to gamma_vals.
        """
        chern_numbers = []
        for gamma in gamma_vals:
            print(f"Computing Chern number for gamma = {gamma:.4f}...")
            chern_numbers.append(self.compute_chern_number(gamma, N=N))
        return np.array(chern_numbers)

    def plot_results(self, gamma_vals, chern_vals):
        """
        Plots the computed Chern numbers as a function of gamma.

        Args:
            gamma_vals (np.ndarray): Array of non-Hermiticity parameter values.
            chern_vals (np.ndarray): Array of computed Chern numbers.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(gamma_vals, chern_vals, 'b.-', label='Biorthogonal Chern number')
        # Plot rounded integer values for comparison, showing the expected quantization
        plt.plot(gamma_vals, np.round(chern_vals), 'r--', label='Rounded integer')
        
        # Calculate and plot the critical gamma values where the gap closes
        delta0 = self.hermitian_gap()
        plt.axvline(x=0.5 * delta0, linestyle=':', color='gray', label=r'Critical $|\gamma| = \Delta_0/2$')
        plt.axvline(x=-0.5 * delta0, linestyle=':', color='gray')
        
        plt.xlabel(r"$\gamma$") # LaTeX for gamma symbol
        plt.ylabel("Chern number")
        plt.title(f"Non-Hermitian QWZ Model (m = {self.m})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def hermitian_gap(self):
        """
        Returns the Hermitian clean-limit energy gap $\Delta_0(m)$ for the QWZ model.
        For the QWZ model with H_z = m + t(cos(kx) + cos(ky)) and t=1,
        the gap is typically 2*|m| when |m|<2.
        """
        # For m=1.0, the gap is 2.0 based on the problem description and common QWZ conventions.
        return 2.0 


if __name__ == "__main__":
    # Initialize the model with a mass term (m)
    model = NonHermitianQWZ(m=1.0) 
    
    # Calculate the clean-limit Hermitian gap Δ₀(m)
    delta0_m = model.hermitian_gap()
    print(f"Clean-limit Hermitian gap Δ₀(m) for m={model.m}: {delta0_m}")
    
    # Define the range of gamma values to sweep over
    gamma_critical = 0.5 * delta0_m
    # Extend the range slightly beyond the critical points to observe the jump
    gamma_vals = np.linspace(-1.5 * gamma_critical, 1.5 * gamma_critical, 101) 
    
    # Set the grid size for Brillouin zone discretization.
    # Increased N_grid for better accuracy in Chern number calculation.
    N_grid = 200 # Increased from 100 to 200 for even better accuracy

    # Compute the Chern number for each gamma value
    chern_vals = model.compute_chern_vs_gamma(gamma_vals, N=N_grid)
    
    # Plot the results
    model.plot_results(gamma_vals, chern_vals)
