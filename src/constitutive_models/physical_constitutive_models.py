from typing import *

import torch
import torch.nn as nn
from torch import Tensor

from .abstract import Elasticity

# --- numerics hardening helpers ---
def eigvalsh_fp32(M: torch.Tensor, chunk: int = 8192, cpu_fallback_free_mb: int = 128) -> torch.Tensor:
    M32 = M.to(torch.float32)
    N = M32.shape[0]
    device = M.device
    
    if device.type == "cuda":
        free_mb = torch.cuda.mem_get_info()[0] // (1024*1024)
        if free_mb < cpu_fallback_free_mb:
            # CPU fallback with error handling
            try:
                out = torch.linalg.eigvalsh(M32.cpu()).to(device)
                return out
            except RuntimeError:
                # Fallback failed, continue to chunked processing
                pass

    # 🔑 preallocate to avoid list-of-tensors (fragmentation)
    out = torch.empty((N, 3), device=device, dtype=M32.dtype)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        # compute into a temporary then copy_ into preallocated slice
        try:
            tmp = torch.linalg.eigvalsh(M32[i:j])
        except RuntimeError as e:
            # Add regularization for ill-conditioned matrices
            eps_reg = 1e-6
            M_reg = M32[i:j] + eps_reg * torch.eye(3, device=device, dtype=M32.dtype).unsqueeze(0)
            try:
                tmp = torch.linalg.eigvalsh(M_reg)
            except RuntimeError:
                # Fallback: use diagonal values
                tmp = torch.diagonal(M32[i:j], dim1=-2, dim2=-1)
        out[i:j].copy_(tmp)
        del tmp
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out

class SigmaElasticity(Elasticity):
    out_measure = "kirchhoff"
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1)
            
        # warp svd
        U, sigma, Vh = self.svd(F)
        threshold = 1e-3
        sigma = torch.clamp_min(sigma, threshold)
        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + la * trace
        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), self.transpose(U))
        return stress
    
    def name(self):
        return "SigmaElasticity"

class CorotatedElasticity(Elasticity):
    out_measure = "kirchhoff"
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)
        # warp svd
        U, sigma, Vh = self.svd(F)
        
        corotated_stress = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        assert torch.all(torch.isfinite(J))
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress
        assert torch.all(torch.isfinite(stress))
        return stress
    
    def name(self):
        return "CorotatedElasticity"
    
class FluidElasticity(Elasticity):
    out_measure = "kirchhoff"
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu

        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)
        
        # Safely set tensor 0 to avoid dtype/device broadcast confusion
        mu = torch.zeros_like(la)
        # warp svd
        U, sigma, Vh = self.svd(F)
        
        corotated_stress = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        assert torch.all(torch.isfinite(J))
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress
        assert torch.all(torch.isfinite(stress))
        return stress
    
    def name(self):
        return "FluidElasticity"
    
class StVKElasticity(Elasticity):
    out_measure = "piola"
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
        
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)

        # warp svd
        U, sigma, Vh = self.svd(F)

        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        E = 0.5 * (FtF - I)

        stvk_stress = 2 * mu * torch.matmul(F, E)

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        volume_stress = la * J * (J - 1) * I

        stress = stvk_stress + volume_stress

        return stress
    
    def name(self):
        return "StVKElasticity"
    
    
class VolumeElasticity(Elasticity):
    out_measure = "kirchhoff"
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))


        self.mode = 'taichi'

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)

        J = torch.det(F).view(-1, 1, 1)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)

        if self.mode.casefold() == 'ziran':

            #  https://en.wikipedia.org/wiki/Bulk_modulus
            kappa = 2 / 3 * mu + la

            # https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/ConstitutiveModel/EquationOfState.h
            # using gamma = 7 would have gradient issue, fix later
            gamma = 2

            stress = kappa * (J - 1 / torch.pow(J, gamma-1)) * I

        elif self.mode.casefold() == 'taichi':

            stress = la * J * (J - 1) * I

        else:
            raise ValueError('invalid mode for volume plasticity: {}'.format(self.mode))

        return stress
    
    def name(self):
        return "VolumeElasticity"
    
    
# ====================================================================
# ================= NEW FRACTURE MODELS ADDED BELOW ==================
# ====================================================================

class BrittleFractureElasticity(Elasticity):
    out_measure = "piola"
    """
    A simple brittle fracture model.
    If the maximum principal stress exceeds the fracture_threshold, the stress is set to zero.
    Suitable for simple simulations of brittle materials like concrete or stone.
    """
    def __init__(self) -> None:
        super().__init__()

        # Basic elastic properties
        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))
        # New parameter: the maximum tensile stress the material can withstand
        self.register_buffer('fracture_threshold', torch.Tensor([1.0e4]))

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
        
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1, 1)
        if la.dim() != 0:
            la = la.reshape(-1, 1, 1)

        # 1. Calculate the stress of the undamaged material (based on StVK model logic)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)
        E_strain = 0.5 * (FtF - I) # Green-Lagrange strain tensor
        S = la * torch.einsum('bii->b', E_strain).view(-1, 1, 1) * I + 2 * mu * E_strain # 2nd Piola-Kirchhoff stress
        
        # 🛡️ FIX: F @ S is Piola P (comment corrected)
        P_undamaged = torch.matmul(F, S)  # <- Piola (comment corrected)

        # 2. Brittle determination proceeds from Cauchy (this part remains unchanged)
        J = torch.det(F).view(-1, 1, 1).clamp_min(1e-8)
        sigma = (1.0 / J) * torch.matmul(P_undamaged, self.transpose(F))
        sigma_sym = 0.5 * (sigma + self.transpose(sigma))
        with torch.no_grad():
            principal_stresses = eigvalsh_fp32(sigma_sym, chunk=getattr(self, "eig_chunk", 8192))
        max_principal_stress = principal_stresses[..., -1]

        # 3. Check the fracture condition
        is_fractured = (max_principal_stress > self.fracture_threshold).view(-1, 1, 1)

        # 4. If brittle, stress is 0, otherwise maintain original Piola
        P_final = torch.where(is_fractured, torch.zeros_like(P_undamaged), P_undamaged)

        # 5. Since out_measure = "piola", return as is (no inverse needed!)
        return P_final
    
    def name(self):
        return "BrittleFractureElasticity"


class PhaseFieldElasticity(Elasticity):
    out_measure = "kirchhoff"
    """
    Phase-field fracture model (tension-degraded).
    Returns Kirchhoff stress tau for mixing stability.
    """
    # ↓ This flag is essential to prevent duplicate external damping
    use_phase_field = True
    
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))
        self.register_buffer('k', torch.Tensor([1e-9]))

    # Put e_cat first, then c. (Compatible with main.py calls)
    def forward(
        self,
        F: Tensor,
        e_cat: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        log_E: Optional[Tensor] = None,
        nu: Optional[Tensor] = None,
    ) -> Tensor:
        # If c is None, set to 0 (for compatibility)
        if c is None:
            c = torch.zeros(F.shape[0], 1, 1, device=F.device, dtype=F.dtype)
        elif c.dim() == 1:
            c = c[:, None, None]
        elif c.dim() == 2:
            c = c[:, :, None]
        c = c.to(device=F.device, dtype=F.dtype)

        E_mod = (self.log_E if log_E is None else log_E).exp()
        nu = self.nu if nu is None else nu

        mu = E_mod / (2 * (1 + nu))
        lam = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
        if mu.dim() != 0:  mu = mu.reshape(-1, 1, 1)
        if lam.dim() != 0: lam = lam.reshape(-1, 1, 1)

        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        C = F.transpose(1, 2) @ F
        E_gl = 0.5 * (C - I)

        # spectral split E = E+ + E- without eigenvectors
        # compute only eigenvalues with low-memory path and no autograd
        with torch.no_grad():
            eig = eigvalsh_fp32(E_gl, chunk=getattr(self, "eig_chunk", 8192))  # (N,3) sorted ascending
        eig_plus  = torch.clamp_min(eig, 0.0)
        eig_minus = torch.clamp_max(eig, 0.0)

        # build spectral projectors using Cayley-Hamilton polynomials
        # Pk = prod_{j!=k} (E - eig_j I) / prod_{j!=k} (eig_k - eig_j)
        e0 = eig[:, 0].view(-1, 1, 1)
        e1 = eig[:, 1].view(-1, 1, 1)
        e2 = eig[:, 2].view(-1, 1, 1)
        A = E_gl
        A_e0 = A - e0 * I
        A_e1 = A - e1 * I
        A_e2 = A - e2 * I
        # denominators with small eps for degeneracy
        eps = 1e-8
        d0 = (eig[:, 0] - eig[:, 1]) * (eig[:, 0] - eig[:, 2])
        d1 = (eig[:, 1] - eig[:, 0]) * (eig[:, 1] - eig[:, 2])
        d2 = (eig[:, 2] - eig[:, 0]) * (eig[:, 2] - eig[:, 1])
        d0 = d0.view(-1, 1, 1) + eps
        d1 = d1.view(-1, 1, 1) + eps
        d2 = d2.view(-1, 1, 1) + eps
        P0 = (A_e1 @ A_e2) / d0
        P1 = (A_e0 @ A_e2) / d1
        P2 = (A_e0 @ A_e1) / d2

        E_plus  = eig_plus[:, 0].view(-1, 1, 1)  * P0 \
                + eig_plus[:, 1].view(-1, 1, 1)  * P1 \
                + eig_plus[:, 2].view(-1, 1, 1)  * P2
        E_minus = eig_minus[:, 0].view(-1, 1, 1) * P0 \
                + eig_minus[:, 1].view(-1, 1, 1) * P1 \
                + eig_minus[:, 2].view(-1, 1, 1) * P2

        tr_plus  = torch.einsum('bii->b', E_plus ).view(-1, 1, 1)
        tr_minus = torch.einsum('bii->b', E_minus).view(-1, 1, 1)

        # 2nd PK split
        S_plus  = lam * tr_plus  * I + 2 * mu * E_plus
        S_minus = lam * tr_minus * I + 2 * mu * E_minus

        # 🆕 Enhanced stiffness degradation by cracks: higher damage leads to greater stiffness reduction
        
        # make exponent configurable; default=3
        exp = getattr(self, "damage_exp", 3)
        g = (1.0 - c).pow(exp) + self.k
        
        # Set stiffness to nearly zero if damage exceeds threshold
        damage_threshold = 0.8
        mask_cracked = c.squeeze(-1) > damage_threshold
        g[mask_cracked] = self.k  # Cracked parts maintain only minimum stiffness
        
        # return Kirchhoff tau = F S F^T
        tau = F @ (g * S_plus + S_minus) @ F.transpose(1, 2)
        tau = torch.nan_to_num(tau, 0.0, 0.0, 0.0).clamp(-1e8, 1e8)
        return tau

    def energy_density(self, F: Tensor, e_cat: Optional[Tensor] = None) -> Tensor:
        """
        Compute elastic energy density for phase-field damage evolution.
        Args:
            F: deformation gradient (N, 3, 3)
            e_cat: elastic parameters (ignored for physical models)
        Returns:
            energy density (N,)
        """
        # normalize c -> (N,1,1) - use zero damage for energy computation
        c = torch.zeros(F.shape[0], 1, 1, device=F.device, dtype=F.dtype)
        
        E_mod = self.log_E.exp()
        nu = self.nu

        mu = E_mod / (2 * (1 + nu))
        lam = E_mod * nu / ((1 + nu) * (1 - 2 * nu))

        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        C = F.transpose(1, 2) @ F
        E_gl = 0.5 * (C - I)

        # compute energy density from strain
        tr_E = torch.einsum('bii->b', E_gl).view(-1, 1, 1)
        E_dot_E = torch.einsum('bij,bij->b', E_gl, E_gl).view(-1, 1, 1)
        
        # elastic energy density: (1/2) * (lam * tr(E)^2 + 2*mu * E:E)
        energy = 0.5 * (lam * tr_E**2 + 2 * mu * E_dot_E)
        return energy.view(-1)   # (N,)
    
    def name(self):
        return "PhaseFieldElasticity"


class CorotatedPhaseFieldElasticity(Elasticity):
    """
    Corotated elasticity with phase field damage degradation.

    Uses SVD-based spectral split (numerically stable for large deformations)
    instead of Green-Lagrange strain (SVK-like, which diverges).

    Math:
        F = U @ diag(sigma) @ Vh           (SVD)
        epsilon_i = sigma_i - 1            (principal strains)
        Tension/compression split on epsilon
        g(c) = (1-c)^exp + k              (degradation, tension only)
        tau = U @ diag(P_hat) @ U^T        (Kirchhoff stress)
    """
    out_measure = "kirchhoff"
    use_phase_field = True

    def __init__(self, Gc: float = 50.0, l0: float = 0.03) -> None:
        super().__init__()
        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))
        self.register_buffer('k', torch.Tensor([1e-9]))  # Residual stiffness
        # Phase field fracture parameters (AT2 model)
        self.Gc = Gc    # Fracture toughness (J/m²) — energy needed to fully crack
        self.l0 = l0    # Regularization length — controls crack width (~2-3x dx)

    def forward(
        self,
        F: Tensor,
        e_cat: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        log_E: Optional[Tensor] = None,
        nu: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Kirchhoff stress with corotated spectral split + damage.

        Args:
            F: (N, 3, 3) deformation gradient
            e_cat: ignored (compatibility with data-driven models)
            c: (N,) or (N,1) or (N,1,1) damage field [0,1]
            log_E: optional log Young's modulus override
            nu: optional Poisson's ratio override

        Returns:
            tau: (N, 3, 3) Kirchhoff stress
        """
        N = F.shape[0]

        # 1. Normalize c to (N, 1)
        if c is None:
            c = torch.zeros(N, 1, device=F.device, dtype=F.dtype)
        elif c.dim() == 1:
            c = c[:, None]
        while c.dim() > 2:
            c = c.squeeze(-1)
        c = c.to(device=F.device, dtype=F.dtype)

        # 2. Material parameters
        E_mod = (self.log_E if log_E is None else log_E).exp()
        nu_val = self.nu if nu is None else nu
        mu = E_mod / (2 * (1 + nu_val))
        lam = E_mod * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

        # 3. SVD of F
        U, sigma, Vh = self.svd(F)  # sigma: (N, 3)
        sigma = torch.clamp_min(sigma, 1e-6)

        # 4. Principal strains
        epsilon = sigma - 1.0  # (N, 3)

        # 5. Spectral split
        eps_plus = torch.clamp_min(epsilon, 0.0)   # tension
        eps_minus = torch.clamp_max(epsilon, 0.0)   # compression

        tr_eps = epsilon.sum(dim=1, keepdim=True)    # (N, 1)
        tr_plus = torch.clamp_min(tr_eps, 0.0)
        tr_minus = torch.clamp_max(tr_eps, 0.0)

        # 6. Energy derivatives w.r.t. singular values
        dPsi_plus = 2.0 * mu * eps_plus + lam * tr_plus      # (N, 3)
        dPsi_minus = 2.0 * mu * eps_minus + lam * tr_minus    # (N, 3)

        # 7. Degradation function g(c)
        exp = getattr(self, "damage_exp", 3)
        g = (1.0 - c).pow(exp) + self.k  # (N, 1)

        # Hard cutoff for heavily damaged particles
        mask_cracked = c.squeeze(-1) > 0.8
        g[mask_cracked] = self.k

        # 8. Combine: degraded tension + undegraded compression
        P_hat = (g * dPsi_plus + dPsi_minus) * sigma  # (N, 3)

        # 9. Kirchhoff stress: tau = U @ diag(P_hat) @ U^T
        tau = torch.matmul(
            torch.matmul(U, torch.diag_embed(P_hat)),
            self.transpose(U)
        )

        tau = torch.nan_to_num(tau, 0.0, 0.0, 0.0).clamp(-1e8, 1e8)
        return tau

    def tension_energy_density(self, F: Tensor) -> Tensor:
        """
        Compute tension-only energy density (for phase field driving force).
        Uses undegraded material (c=0).

        psi_plus = mu * sum(eps_plus_i^2) + lam/2 * max(tr(eps), 0)^2

        Args:
            F: (N, 3, 3) deformation gradient
        Returns:
            psi_plus: (N,) tension energy density
        """
        E_mod = self.log_E.exp()
        nu_val = self.nu
        mu = E_mod / (2 * (1 + nu_val))
        lam = E_mod * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

        U, sigma, Vh = self.svd(F)
        sigma = torch.clamp_min(sigma, 1e-6)
        epsilon = sigma - 1.0

        eps_plus = torch.clamp_min(epsilon, 0.0)
        tr_eps = epsilon.sum(dim=1, keepdim=True)
        tr_plus = torch.clamp_min(tr_eps, 0.0)

        psi_plus = mu * (eps_plus ** 2).sum(dim=1) + 0.5 * lam * tr_plus.squeeze(-1) ** 2
        return psi_plus.clamp(min=0.0)  # (N,)

    def energy_density(self, F: Tensor, e_cat: Optional[Tensor] = None) -> Tensor:
        """
        Compute total elastic energy density (undegraded).

        Args:
            F: (N, 3, 3) deformation gradient
            e_cat: ignored
        Returns:
            energy: (N,) total energy density
        """
        E_mod = self.log_E.exp()
        nu_val = self.nu
        mu = E_mod / (2 * (1 + nu_val))
        lam = E_mod * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

        U, sigma, Vh = self.svd(F)
        sigma = torch.clamp_min(sigma, 1e-6)
        epsilon = sigma - 1.0

        tr_eps = epsilon.sum(dim=1, keepdim=True)
        energy = mu * (epsilon ** 2).sum(dim=1) + 0.5 * lam * tr_eps.squeeze(-1) ** 2
        return energy.clamp(min=0.0)  # (N,)

    def name(self):
        return "CorotatedPhaseFieldElasticity"


def piola_to_kirchhoff(F: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Convert Piola stress to Kirchhoff stress: τ = (1/J) * P * F^T
    
    Args:
        F: deformation gradient (N, 3, 3)
        P: Piola stress (N, 3, 3)
    
    Returns:
        τ: Kirchhoff stress (N, 3, 3)
    """
    J = torch.det(F).view(-1, 1, 1).clamp_min(1e-8)
    return (P @ F.transpose(1, 2)) / J