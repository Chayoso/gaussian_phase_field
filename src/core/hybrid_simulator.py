"""
Hybrid Crack Simulator

Main integration: MPM physics + Gaussian Splatting visualization

Orchestrates:
- MPM particle-based physics simulation
- Phase field crack propagation
- Volumetric-to-surface damage projection
- Gaussian Splat rendering updates
"""

import torch
from torch import Tensor
from typing import Dict, Optional
import time
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.mpm_core.mpm_model import MPMModel
from src.constitutive_models.phase_field import update_phase_field
from src.constitutive_models.damage_mapper import VolumetricToSurfaceDamageMapper
from src.visualization.gaussian_updater import GaussianCrackVisualizer
from src.core.coordinate_mapper import CoordinateMapper


class HybridCrackSimulator:
    """
    Unified MPM + Gaussian Splats simulator for crack visualization

    Data flow per timestep:
    1. MPM physics: (x, v, F, c) → p2g2p → (x', v', F', c')
    2. Coordinate map: x_mpm' → x_world'
    3. Damage project: c_vol' → c_surf'
    4. Gaussian update: positions, opacity, color
    5. Rendering: gaussians → render() → image
    """

    def __init__(
        self,
        mpm_model: MPMModel,
        gaussians,  # GaussianModel instance
        elasticity_module,  # PhaseFieldElasticity or CorotatedPhaseFieldElasticity
        coord_mapper: CoordinateMapper,
        damage_mapper: VolumetricToSurfaceDamageMapper,
        visualizer: GaussianCrackVisualizer,
        surface_mask: Tensor,
        physics_substeps: int = 10,
        phase_field_params: Optional[Dict] = None,
        simulation_mode: str = "crack_only",
        seismic_params: Optional[Dict] = None
    ):
        """
        Args:
            mpm_model: MPM simulation model
            gaussians: Gaussian Splatting model
            elasticity_module: Constitutive model (e.g., CorotatedPhaseFieldElasticity)
            coord_mapper: MPM ↔ world coordinate transformer
            damage_mapper: Volumetric → surface damage projector
            visualizer: Gaussian property updater
            surface_mask: (N_mpm,) boolean mask for surface particles
            physics_substeps: MPM steps per render frame
            phase_field_params: Parameters for update_phase_field()
            simulation_mode: "crack_only" (Fisher-KPP) or "deformation" (MPM + phase field)
        """
        self.mpm = mpm_model
        self.gaussians = gaussians
        self.elasticity = elasticity_module
        self.mapper = coord_mapper
        self.damage_mapper = damage_mapper
        self.visualizer = visualizer
        self.surface_mask = surface_mask
        self.substeps = physics_substeps
        self.pf_params = phase_field_params or {}
        self.simulation_mode = simulation_mode

        # Seismic loading parameters
        self.seismic = seismic_params or {}
        self.seismic_enabled = self.seismic.get("enabled", False)

        # Simulation state tensors
        self.x_mpm = None    # (N_mpm, 3) MPM positions [0, 1]³
        self.v_mpm = None    # (N_mpm, 3) velocities
        self.F = None        # (N_mpm, 3, 3) deformation gradients
        self.C = None        # (N_mpm, 3, 3) affine velocity
        self.c_vol = None    # (N_mpm,) volumetric damage

        # Crack-only mode (no deformation, Fisher-KPP reaction-diffusion)
        self.crack_only = (simulation_mode == "crack_only")
        self.psi_static = None  # Static strain energy field from impact

        # Statistics
        self.frame_count = 0
        self.last_render_time = time.time()
        self.init_positions = None  # For divergence detection

        device = next(mpm_model.parameters()).device if hasattr(mpm_model, 'parameters') else torch.device('cuda')

        print(f"\n{'='*60}")
        print(f"HybridCrackSimulator Initialized")
        print(f"{'='*60}")
        print(f"  - Simulation mode: {simulation_mode}")
        print(f"  - MPM particles: {self.surface_mask.shape[0]}")
        print(f"  - Surface particles: {self.surface_mask.sum().item()}")
        print(f"  - Physics substeps: {physics_substeps}")
        if self.seismic_enabled:
            print(f"  - Seismic loading: ON")
            print(f"    amplitude={self.seismic.get('amplitude')}, "
                  f"freq={self.seismic.get('frequency')}Hz, "
                  f"dir={self.seismic.get('direction')}")
        print(f"  - Device: {device}")
        print(f"{'='*60}\n")

    def initialize(self, init_positions: Tensor):
        """
        Initialize simulation state

        Args:
            init_positions: (N_mpm, 3) initial MPM positions in [0, 1]³
        """
        device = init_positions.device
        N = init_positions.shape[0]

        print(f"[HybridSimulator] Initializing state...")
        print(f"  - Particles: {N}")
        print(f"  - Device: {device}")

        # Initialize MPM state
        self.x_mpm = init_positions.clone()
        self.v_mpm = torch.zeros((N, 3), device=device)
        self.F = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3).clone()
        self.C = torch.zeros((N, 3, 3), device=device)
        self.c_vol = torch.zeros(N, device=device)

        # Store initial positions for divergence detection
        self.init_positions = init_positions.clone()

        # Initialize Gaussian positions (surface only)
        x_surf_mpm = self.x_mpm[self.surface_mask]
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)

        # Update Gaussian positions
        self.gaussians._xyz.data = x_surf_world

        # Reset frame counter
        self.frame_count = 0
        self.last_render_time = time.time()

        print(f"  - State initialized")
        print(f"  - Surface Gaussians: {x_surf_world.shape[0]}")

    def initialize_crack_energy(
        self,
        impact_center_mpm: Tensor,
        impact_energy: float = 1.0,
        impact_radius: float = 0.03
    ):
        """
        Create grid-based damage seed and enable crack-only mode.

        Seeds a Gaussian blob of damage directly on the 3D grid at the impact
        point. Also creates an occupancy mask to constrain crack propagation
        to the object's volume. The Fisher-KPP reaction-diffusion then
        propagates the crack wavefront through the object.

        Args:
            impact_center_mpm: (3,) impact point in MPM space [0,1]^3
            impact_energy: Peak energy (unused, seed is always 1.0)
            impact_radius: Gaussian decay radius in MPM space
        """
        n = self.mpm.num_grids
        dx = self.mpm.dx
        device = self.x_mpm.device
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P

        # 1. Create persistent grid damage field seeded at impact point
        coords_1d = torch.arange(n, device=device).float() * dx
        gi, gj, gk = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        grid_pos = torch.stack([gi, gj, gk], dim=-1)  # (n,n,n,3)

        dists = (grid_pos - impact_center_mpm.view(1, 1, 1, 3)).norm(dim=-1)
        self.c_grid = torch.exp(-dists ** 2 / (2 * impact_radius ** 2))
        self.c_grid = self.c_grid.clamp(0.0, 1.0)

        # 2. Create occupancy mask from particle positions
        grid_w = torch.zeros(n ** 3, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            grid_w.index_add_(0, index.reshape(-1), weight.reshape(-1))
        self.grid_occupied = (grid_w.view(n, n, n) > 1e-6)

        # Dilate occupancy by 1 cell so crack can reach object edges
        occ = self.grid_occupied.float().unsqueeze(0).unsqueeze(0)
        occ_dilated = torch.nn.functional.max_pool3d(
            occ, kernel_size=3, stride=1, padding=1
        )
        self.grid_occupied = (occ_dilated[0, 0] > 0)

        # Mask initial seed to object volume only
        self.c_grid = self.c_grid * self.grid_occupied.float()

        # Store seed for persistent re-application (keeps impact at c=1.0)
        self.c_grid_seed = self.c_grid.clone()

        # Gather initial damage to particles
        self._gather_grid_to_particles()

        # Statistics
        n_seeded = (self.c_grid > 0.01).sum().item()
        n_high = (self.c_grid > 0.5).sum().item()
        n_occupied = self.grid_occupied.sum().item()
        print(f"\n[CrackOnly] Grid-based crack initialized:")
        print(f"  - Impact center (MPM): {impact_center_mpm.detach().cpu().numpy()}")
        print(f"  - Impact radius: {impact_radius}")
        print(f"  - Grid seeded cells (c>0.01): {n_seeded}")
        print(f"  - Grid high cells (c>0.5): {n_high}")
        print(f"  - Grid occupied cells: {n_occupied}/{n ** 3}")
        print(f"  - c_grid max: {self.c_grid.max():.4f}")
        print(f"  - c_vol max (particles): {self.c_vol.max():.4f}")
        print(f"  - Crack-only mode ENABLED (grid-based Fisher-KPP)")

    def initialize_deformation_impact(
        self,
        impact_center_mpm: Tensor,
        impact_energy: float = 1.0,
        impact_radius: float = 0.03,
        impact_direction: Optional[Tensor] = None
    ):
        """
        Apply impact for deformation mode: inward velocity impulse + surface damage seed.

        The impulse is directed inward (camera → object), creating compression
        at the impact site and tension behind it. Damage is seeded only on
        surface particles near the impact point, so the crack starts at the
        surface and propagates inward via the phase field.

        Args:
            impact_center_mpm: (3,) impact point in MPM space [0,1]^3
            impact_energy: Impulse magnitude (velocity scale)
            impact_radius: Gaussian decay radius in MPM space
            impact_direction: (3,) normalized direction of impact (camera→object)
        """
        device = self.x_mpm.device

        # Distance from impact
        dists = (self.x_mpm - impact_center_mpm.unsqueeze(0)).norm(dim=1)
        influence = torch.exp(-dists ** 2 / (2 * impact_radius ** 2))

        # 1. Inward velocity impulse (camera → object direction)
        if impact_direction is not None:
            # Directed impulse: all affected particles pushed in same direction (inward)
            imp_dir = impact_direction.to(device)
            imp_dir = imp_dir / (imp_dir.norm() + 1e-12)
            self.v_mpm = self.v_mpm + imp_dir.unsqueeze(0) * influence.unsqueeze(1) * impact_energy
        else:
            # Fallback: radial inward (toward impact center)
            directions = impact_center_mpm.unsqueeze(0) - self.x_mpm
            directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
            self.v_mpm = self.v_mpm + directions * influence.unsqueeze(1) * impact_energy

        # 2. Damage seed on particles (volumetric pre-crack for stress degradation)
        self.c_vol = torch.maximum(self.c_vol, influence * 0.8)

        # 3. Initialize grid for hybrid Fisher-KPP propagation
        n = self.mpm.num_grids
        dx = self.mpm.dx
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P

        # Seed grid damage at impact (same as crack_only mode)
        coords_1d = torch.arange(n, device=device).float() * dx
        gi, gj, gk = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        grid_pos = torch.stack([gi, gj, gk], dim=-1)
        grid_dists = (grid_pos - impact_center_mpm.view(1, 1, 1, 3)).norm(dim=-1)
        self.c_grid = torch.exp(-grid_dists ** 2 / (2 * impact_radius ** 2)).clamp(0.0, 1.0)

        # Occupancy mask from particle positions
        grid_w = torch.zeros(n ** 3, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            grid_w.index_add_(0, index.reshape(-1), weight.reshape(-1))
        self.grid_occupied = (grid_w.view(n, n, n) > 1e-6)

        # Dilate occupancy by 1 cell
        occ = self.grid_occupied.float().unsqueeze(0).unsqueeze(0)
        occ_dilated = torch.nn.functional.max_pool3d(occ, kernel_size=3, stride=1, padding=1)
        self.grid_occupied = (occ_dilated[0, 0] > 0)

        # Mask to object volume
        self.c_grid = self.c_grid * self.grid_occupied.float()
        self.c_grid_seed = self.c_grid.clone()

        # Initialize H_grid for stress-modulated propagation
        self.H_grid = torch.zeros(n, n, n, device=device)

        # Statistics
        n_impulse = (influence > 0.01).sum().item()
        n_total_damaged = (self.c_vol > 0.001).sum().item()
        n_grid_seeded = (self.c_grid > 0.01).sum().item()
        n_occupied = self.grid_occupied.sum().item()
        v_max = self.v_mpm.abs().max().item()
        print(f"\n[Deformation+Hybrid] Impact applied:")
        print(f"  - Impact center (MPM): {impact_center_mpm.detach().cpu().numpy()}")
        print(f"  - Impact direction: {'camera→object' if impact_direction is not None else 'radial inward'}")
        print(f"  - Impact energy: {impact_energy}, radius: {impact_radius}")
        print(f"  - Particles with impulse (>0.01): {n_impulse}")
        print(f"  - Total particles with damage: {n_total_damaged}")
        print(f"  - Grid seeded cells: {n_grid_seeded}/{n**3}")
        print(f"  - Grid occupied cells: {n_occupied}/{n**3}")
        print(f"  - Max velocity: {v_max:.4f}")
        print(f"  - Max damage seed: {self.c_vol.max():.4f}")
        print(f"  - Hybrid mode: MPM physics + grid Fisher-KPP + H modulation")

    @torch.no_grad()
    def step_crack_only(self, dt: float):
        """
        Phase field evolution without deformation (crack-only mode).

        Runs Fisher-KPP reaction-diffusion directly on the persistent c_grid:
            dc/dt = D * lap(c) + alpha * c * (1 - c)

        This creates a propagating wavefront with speed v = 2*sqrt(D*alpha).
        The grid is seeded at impact by initialize_crack_energy() and the
        wavefront propagates outward through occupied cells only.

        Args:
            dt: Physics time step (not directly used; crack has its own dt)
        """
        if not hasattr(self, '_crack_step'):
            self._crack_step = 0

        # Parameters
        diff_coeff = self.pf_params.get('crack_diff_coeff', 0.1)
        alpha = self.pf_params.get('crack_alpha', 10.0)
        n_iters = self.pf_params.get('crack_grid_iters', 10)

        n = self.mpm.num_grids
        dx = self.mpm.dx

        # Stability limit for explicit diffusion: dt < dx^2 / (2*D*3)
        dt_rd = 0.8 * dx * dx / (2.0 * diff_coeff * 3.0 + 1e-12)
        dt_rd = min(dt_rd, 0.01)

        # Run Fisher-KPP iterations on the persistent grid
        occ_float = self.grid_occupied.float()
        for _ in range(n_iters):
            g5 = self.c_grid.unsqueeze(0).unsqueeze(0)
            gp = torch.nn.functional.pad(g5, (1,1,1,1,1,1), mode='replicate')[0, 0]
            lap = (
                gp[2:, 1:-1, 1:-1] + gp[:-2, 1:-1, 1:-1] +
                gp[1:-1, 2:, 1:-1] + gp[1:-1, :-2, 1:-1] +
                gp[1:-1, 1:-1, 2:] + gp[1:-1, 1:-1, :-2] - 6.0 * self.c_grid
            ) / (dx * dx)

            reaction = alpha * self.c_grid * (1.0 - self.c_grid)
            self.c_grid = self.c_grid + dt_rd * (diff_coeff * lap + reaction)
            self.c_grid = self.c_grid.clamp(0.0, 1.0)

            # Constrain to object volume (prevent leaking into empty space)
            self.c_grid = self.c_grid * occ_float

            # Note: no re-seeding. The Fisher-KPP reaction sustains the wavefront.
            # The blob must be large enough (radius >> sqrt(D/alpha)) to be self-sustaining.

        # Gather from grid to particles
        c_old = self.c_vol.clone()
        self._gather_grid_to_particles()

        # Irreversible: damage can only increase
        self.c_vol = torch.maximum(self.c_vol, c_old)

        # Diagnostics
        step = self._crack_step
        if step < 20 or step % 10 == 0:
            dc = self.c_vol - c_old
            n_01 = (self.c_vol > 0.01).sum().item()
            n_30 = (self.c_vol > 0.3).sum().item()
            n_80 = (self.c_vol > 0.8).sum().item()
            gc_max = self.c_grid.max().item()
            gc_cells = (self.c_grid > 0.3).sum().item()
            print(f"  [crack {step:3d}] c_max={self.c_vol.max():.4f} "
                  f"gc_max={gc_max:.4f} gc_cells(>0.3)={gc_cells} "
                  f"dc_max={dc.max():.6f} "
                  f"cracked(>0.01)={n_01} "
                  f"cracked(>0.3)={n_30} "
                  f"cracked(>0.8)={n_80}", flush=True)

        self._crack_step += 1

    @torch.no_grad()
    def _gather_grid_to_particles(self):
        """
        Interpolate damage from c_grid to particle positions using
        MPM quadratic B-spline weights.
        """
        grid_flat = self.c_grid.reshape(-1)
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P
        device = self.x_mpm.device

        c_new = torch.empty(P, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            c_new[i:j] = (weight * grid_flat[index].view(-1, 27)).sum(dim=1)

        self.c_vol = c_new.clamp(0.0, 1.0)

    @torch.no_grad()
    def _apply_seismic_loading(self, dt: float):
        """
        Apply oscillating seismic ground acceleration to all particles.

        Models earthquake ground motion as sinusoidal body acceleration:
            a(t) = amplitude * sin(2π * freq * t) * direction * envelope(t)

        This creates alternating tension/compression cycles that:
        - Concentrate stress at crack tip (damage seed)
        - Accumulate H (history variable) each cycle
        - Drive AT2 crack propagation progressively

        Args:
            dt: Physics timestep
        """
        if not self.seismic_enabled:
            return

        import math

        t = self.mpm.time  # Current simulation time
        amp = self.seismic.get("amplitude", 1000.0)
        freq = self.seismic.get("frequency", 80.0)
        direction = self.seismic.get("direction", [1.0, 0.0, 0.0])
        ramp_time = self.seismic.get("ramp_time", 0.005)

        device = self.v_mpm.device
        dir_tensor = torch.tensor(direction, device=device, dtype=self.v_mpm.dtype)
        dir_tensor = dir_tensor / (dir_tensor.norm() + 1e-12)

        # Envelope: ramp up over ramp_time to avoid initial shock
        envelope = min(t / ramp_time, 1.0) if ramp_time > 0 else 1.0

        # Sinusoidal acceleration
        accel = amp * math.sin(2.0 * math.pi * freq * t) * envelope

        # Apply as velocity increment to all particles
        self.v_mpm += dir_tensor.unsqueeze(0) * (accel * dt)

    @torch.no_grad()
    def _init_grid_infrastructure(self):
        """
        Initialize grid infrastructure for hybrid Fisher-KPP propagation.
        Called automatically if no impact seed was provided (seismic-only mode).
        Creates empty c_grid, occupancy mask, and H_grid.
        """
        n = self.mpm.num_grids
        device = self.x_mpm.device
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P

        # Empty damage grid (nucleation will seed it)
        self.c_grid = torch.zeros(n, n, n, device=device)

        # Occupancy mask from current particle positions
        grid_w = torch.zeros(n ** 3, device=device)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            grid_w.index_add_(0, index.reshape(-1), weight.reshape(-1))
        self.grid_occupied = (grid_w.view(n, n, n) > 1e-6)

        # Dilate occupancy by 1 cell
        occ = self.grid_occupied.float().unsqueeze(0).unsqueeze(0)
        occ_dilated = torch.nn.functional.max_pool3d(occ, kernel_size=3, stride=1, padding=1)
        self.grid_occupied = (occ_dilated[0, 0] > 0)

        self.c_grid_seed = self.c_grid.clone()
        self.H_grid = torch.zeros(n, n, n, device=device)

        n_occupied = self.grid_occupied.sum().item()
        print(f"\n[Grid Init] Auto-initialized grid infrastructure (seismic-only mode)")
        print(f"  - Grid occupied cells: {n_occupied}/{n**3}")
        print(f"  - c_grid: all zeros (nucleation will seed)")

    @torch.no_grad()
    def _bin_particles_to_grid(self, values: Tensor) -> Tensor:
        """
        Bin particle scalar values to grid using MPM B-spline weights.
        Uses weighted average: grid_val = sum(w_i * val_i) / sum(w_i)

        Args:
            values: (N,) per-particle scalar values

        Returns:
            (n, n, n) grid values (weighted average)
        """
        n = self.mpm.num_grids
        P = self.x_mpm.shape[0]
        chunk = self.mpm.particle_chunk or P
        device = self.x_mpm.device

        grid_num = torch.zeros(n ** 3, device=device)
        grid_den = torch.zeros(n ** 3, device=device)

        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self.mpm._weight_and_index(self.x_mpm[i:j])
            # weight: (batch, 27), index: (batch, 27)
            wv = weight * values[i:j].unsqueeze(1)  # weighted values
            grid_num.index_add_(0, index.reshape(-1), wv.reshape(-1))
            grid_den.index_add_(0, index.reshape(-1), weight.reshape(-1))

        # Weighted average (avoid div by zero)
        grid_val = grid_num / (grid_den + 1e-12)
        return grid_val.view(n, n, n)

    @torch.no_grad()
    def step_hybrid_crack(self, dt: float):
        """
        Hybrid crack propagation: grid-based Fisher-KPP with H-modulated reaction rate.

        The key insight: AT2's Laplacian term (l0²∇²c) is too weak on uniform MPM grids.
        Instead, we use Fisher-KPP's reaction term c(1-c) for propagation, with the
        reaction rate alpha modulated by the physics-computed stress history H.

        This gives us:
        - Sharp wavefront (controlled by D/alpha ratio)
        - Physics-directed propagation (H higher at stress concentrations)
        - Decoupled from AT2's l0 limitation

        Algorithm:
        1. Bin particle _history_H to H_grid (weighted average)
        2. Compute spatially varying alpha: alpha(x) = alpha_base * clamp(H/H_ref, 0, 1)
        3. Run Fisher-KPP: dc/dt = D*∇²c + alpha(x)*c*(1-c)
        4. Gather grid damage back to particles
        5. Irreversibility: c can only increase

        Args:
            dt: Physics timestep (not directly used; Fisher-KPP has its own dt)
        """
        if not hasattr(self, 'c_grid'):
            # Auto-initialize grid infrastructure (seismic-only mode, no impact seed)
            self._init_grid_infrastructure()

        if not hasattr(self, '_hybrid_step'):
            self._hybrid_step = 0

        n = self.mpm.num_grids
        dx = self.mpm.dx
        device = self.x_mpm.device

        # --- Parameters ---
        D = self.pf_params.get('crack_diff_coeff', 0.0005)     # Low D → sharp front
        alpha_base = self.pf_params.get('crack_alpha', 100.0)   # High alpha → fast propagation
        n_iters = self.pf_params.get('crack_grid_iters', 1)
        Gc = getattr(self.elasticity, 'Gc', 30.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)

        # H_ref: threshold above which full propagation speed is allowed
        # H_ref = Gc / (2*l0) is the AT2 critical driving force
        H_ref = Gc / (2.0 * l0)

        # --- 1) Bin particle H to grid ---
        if hasattr(self, '_history_H'):
            self.H_grid = self._bin_particles_to_grid(self._history_H)
        else:
            self.H_grid = torch.zeros(n, n, n, device=device)

        # --- 2) Compute spatially varying reaction rate ---
        # alpha(x) = alpha_base * clamp(H_grid / H_ref, 0, alpha_max_ratio)
        # Where H > H_ref → full speed, H = 0 → no propagation
        # This ensures crack only grows where stress has been high enough
        H_ratio = (self.H_grid / (H_ref + 1e-12)).clamp(0.0, 2.0)
        alpha_grid = alpha_base * H_ratio  # (n, n, n)

        # --- 2b) Nucleation: seed damage where stress exceeds threshold ---
        # Without this, c=0 → reaction=0, crack never starts.
        # When H > nucleation_fraction * H_ref, nucleate c = c_nucleation.
        nucleation_frac = self.pf_params.get('nucleation_fraction', 0.5)
        c_nucleation = self.pf_params.get('c_nucleation', 0.01)
        nucleation_mask = (self.H_grid > nucleation_frac * H_ref) & (self.c_grid < c_nucleation)
        nucleation_mask = nucleation_mask & self.grid_occupied
        self.c_grid[nucleation_mask] = c_nucleation

        # --- 3) Fisher-KPP iterations on grid ---
        # Stability limit: dt < dx² / (6D)
        dt_rd = 0.8 * dx * dx / (6.0 * D + 1e-12)
        dt_rd = min(dt_rd, 0.01)

        occ_float = self.grid_occupied.float()

        for _ in range(n_iters):
            # 3D Laplacian with replicate boundary
            g5 = self.c_grid.unsqueeze(0).unsqueeze(0)
            gp = torch.nn.functional.pad(g5, (1, 1, 1, 1, 1, 1), mode='replicate')[0, 0]
            lap = (
                gp[2:, 1:-1, 1:-1] + gp[:-2, 1:-1, 1:-1] +
                gp[1:-1, 2:, 1:-1] + gp[1:-1, :-2, 1:-1] +
                gp[1:-1, 1:-1, 2:] + gp[1:-1, 1:-1, :-2] - 6.0 * self.c_grid
            ) / (dx * dx)

            # Fisher-KPP with spatially varying alpha
            reaction = alpha_grid * self.c_grid * (1.0 - self.c_grid)
            self.c_grid = self.c_grid + dt_rd * (D * lap + reaction)
            self.c_grid = self.c_grid.clamp(0.0, 1.0)

            # Constrain to object volume
            self.c_grid = self.c_grid * occ_float

        # --- 4) Gather grid damage to particles ---
        c_old = self.c_vol.clone()
        self._gather_grid_to_particles()

        # --- 5) Irreversibility: damage only increases ---
        self.c_vol = torch.maximum(self.c_vol, c_old)

        # --- 6) Diagnostics ---
        step = self._hybrid_step
        if step < 30 or step % 10 == 0:
            dc = self.c_vol - c_old
            n_growing = (dc > 1e-6).sum().item()
            n_01 = (self.c_vol > 0.01).sum().item()
            n_30 = (self.c_vol > 0.3).sum().item()
            n_80 = (self.c_vol > 0.8).sum().item()
            gc_max = self.c_grid.max().item()
            gc_cells_30 = (self.c_grid > 0.3).sum().item()
            H_max = self.H_grid.max().item()
            H_active = (H_ratio > 0.1).sum().item()
            alpha_max = alpha_grid.max().item()
            print(f"  [hybrid {step:3d}] c_max={self.c_vol.max():.4f} "
                  f"gc_max={gc_max:.4f} gc_cells(>0.3)={gc_cells_30} "
                  f"dc_max={dc.max():.6f} growing={n_growing} "
                  f"H_max={H_max:.2e} H_ref={H_ref:.2e} "
                  f"H_active(>0.1)={H_active} "
                  f"alpha_max={alpha_max:.1f} "
                  f"cracked(>0.01/{n_01} >0.3/{n_30} >0.8/{n_80})",
                  flush=True)

        self._hybrid_step += 1

    @torch.no_grad()
    def step_physics(self, dt: float):
        """
        Single MPM physics timestep (mechanics only, no phase field).

        Updates:
        - Seismic loading (oscillating body force)
        - Stress from elasticity (with damage degradation)
        - MPM P2G2P (particle-grid-particle)
        - Accumulates tension energy H for hybrid crack propagation

        Args:
            dt: Time step (typically mpm.dt ≈ 1e-4)
        """
        # Track physics step count
        if not hasattr(self, '_physics_step'):
            self._physics_step = 0

        # 0. Apply seismic loading (oscillating ground acceleration)
        self._apply_seismic_loading(dt)

        # 1. Compute stress with damage degradation
        stress = self.elasticity(self.F, c=self.c_vol)  # (N, 3, 3)

        # Safety: clamp stress to physical range (E=1e6, max ~5x E)
        stress_limit = 5e6
        stress = stress.clamp(-stress_limit, stress_limit)

        # 2. MPM particle-grid-particle transfer
        x_old = self.x_mpm.clone()
        self.x_mpm, self.v_mpm, self.C, self.F = self.mpm.p2g2p(
            self.x_mpm, self.v_mpm, self.C, self.F, stress
        )

        # Safety: clamp velocity after P2G2P
        v_limit = 5.0
        self.v_mpm = self.v_mpm.clamp(-v_limit, v_limit)

        # 3. Accumulate tension energy H (history variable, irreversible max)
        if hasattr(self.elasticity, 'tension_energy_density'):
            psi = self.elasticity.tension_energy_density(self.F)
        else:
            I = torch.eye(3, device=self.F.device, dtype=self.F.dtype).unsqueeze(0)
            C = self.F.transpose(1, 2) @ self.F
            Egl = 0.5 * (C - I)
            psi = (Egl * Egl).sum(dim=(1, 2))
        # Clamp psi to physical range: max ~ few times Gc/(2*l0) to avoid inf contaminating H
        Gc = getattr(self.elasticity, 'Gc', 30.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)
        psi_max = 10.0 * Gc / (2.0 * l0)  # 10x critical driving force
        psi = torch.clamp(psi, min=0.0, max=psi_max)

        if not hasattr(self, '_history_H'):
            self._history_H = psi.clone()
        else:
            self._history_H = torch.maximum(self._history_H, psi)

        # Per-step diagnostics
        step = self._physics_step
        if step < 50 or step % 10 == 0:
            v_max = self.v_mpm.abs().max().item()
            F_min = self.F.min().item()
            F_max = self.F.max().item()
            s_max = stress.abs().max().item()
            x_min = self.x_mpm.min().item()
            x_max = self.x_mpm.max().item()
            disp = (self.x_mpm - x_old).norm(dim=1).max().item()
            c_wave = (s_max / (self.mpm.p_mass / self.mpm.vol + 1e-12)) ** 0.5
            cfl = c_wave * dt / self.mpm.dx if c_wave > 0 else 0
            print(f"  [step {step:3d}] |v|={v_max:.4f} F=[{F_min:.4f},{F_max:.4f}] "
                  f"|stress|={s_max:.2e} disp={disp:.6f} x=[{x_min:.4f},{x_max:.4f}] "
                  f"CFL~{cfl:.3f}", flush=True)

        self._physics_step += 1

    def step_rendering(self) -> bool:
        """
        Update rendering state (call at visualization framerate, e.g., 30 FPS)

        Performs:
        - Multi-step physics simulation
        - Damage projection to surface
        - Coordinate mapping
        - Gaussian property updates

        Returns:
            True if update successful
        """
        # Multi-step simulation
        if self.crack_only:
            for _ in range(self.substeps):
                self.step_crack_only(self.mpm.dt)
        else:
            # Deformation mode: MPM physics substeps + hybrid crack propagation
            for _ in range(self.substeps):
                self.step_physics(self.mpm.dt)
            # One hybrid crack step per render frame (after all physics substeps)
            self.step_hybrid_crack(self.mpm.dt)
        torch.cuda.empty_cache()  # Free fragmented GPU memory

        # Project volumetric damage to surface particles
        x_surf_mpm = self.x_mpm[self.surface_mask]
        c_surf = self.damage_mapper.project_damage(
            self.c_vol, self.x_mpm, x_surf_mpm, self.surface_mask
        )

        # Map MPM positions to world space
        x_surf_world = self.mapper.mpm_to_world(x_surf_mpm)

        # Update Gaussian Splat properties
        self.visualizer.update_gaussians(
            self.gaussians, c_surf, x_surf_world,
            preserve_original=True
        )

        self.frame_count += 1
        return True

    def apply_external_force(
        self,
        force_center: Tensor,      # (3,) in world space
        force_magnitude: float,
        force_radius: float,
        force_direction: Optional[Tensor] = None,  # (3,) or None (radial)
        surface_only: bool = True   # Only apply to surface particles
    ):
        """
        Apply localized external force to MPM particles

        Args:
            force_center: Force application point (world space)
            force_magnitude: Force strength (impulse)
            force_radius: Influence radius (world space)
            force_direction: Force direction (None = radial outward from center)
            surface_only: If True, only apply force to surface particles
        """
        # Convert force center to MPM space
        force_center_mpm = self.mapper.world_to_mpm(force_center.unsqueeze(0)).squeeze(0)

        # Convert radius to MPM space
        force_radius_mpm = self.mapper.scale_world_to_mpm(force_radius)

        # Compute distances from force center
        dists = (self.x_mpm - force_center_mpm).norm(dim=1)  # (N,)

        # Gaussian falloff
        influence = torch.exp(-dists**2 / (2 * force_radius_mpm**2))  # (N,)

        # Apply surface mask if requested
        if surface_only:
            influence = influence * self.surface_mask.float()  # Zero out interior particles

        # Determine force direction
        if force_direction is None:
            # Radial outward from center
            directions = self.x_mpm - force_center_mpm  # (N, 3)
            directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)
        else:
            # Fixed direction (broadcast)
            force_direction_mpm = force_direction / (force_direction.norm() + 1e-12)
            directions = force_direction_mpm.unsqueeze(0).expand_as(self.x_mpm)

        # Apply impulse to velocities
        impulse = directions * influence.unsqueeze(1) * force_magnitude
        self.v_mpm += impulse

        n_affected = (influence > 0.01).sum().item()
        n_surface_affected = ((influence > 0.01) & self.surface_mask).sum().item() if surface_only else n_affected
        print(f"[HybridSimulator] Applied external force:")
        print(f"  - Center (MPM): {force_center_mpm.detach().cpu().numpy()}")
        print(f"  - Magnitude: {force_magnitude}")
        print(f"  - Surface only: {surface_only}")
        print(f"  - Affected particles: {n_affected}/{self.x_mpm.shape[0]}")
        if surface_only:
            print(f"  - Surface particles affected: {n_surface_affected}")

    def detect_large_deformation(self, threshold: float = 0.3) -> bool:
        """
        Check if particles have moved too far from initial positions

        Args:
            threshold: Maximum displacement as fraction of domain size

        Returns:
            True if large deformation detected
        """
        if self.init_positions is None:
            return False

        displacement = (self.x_mpm - self.init_positions).norm(dim=1)
        max_disp = displacement.max().item()

        return max_disp > threshold

    def get_statistics(self) -> Dict:
        """
        Return current simulation statistics for logging/monitoring

        Returns:
            Dictionary with frame count, damage stats, FPS, etc.
        """
        current_time = time.time()
        elapsed = current_time - self.last_render_time
        fps = 1.0 / (elapsed + 1e-6)
        self.last_render_time = current_time

        stats = {
            "frame": self.frame_count,
            "time": self.mpm.time,
            "c_max": self.c_vol.max().item(),
            "c_mean": self.c_vol.mean().item(),
            "c_surface_max": self.c_vol[self.surface_mask].max().item(),
            "c_surface_mean": self.c_vol[self.surface_mask].mean().item(),
            "n_cracked": (self.c_vol > 0.3).sum().item(),
            "n_particles": self.x_mpm.shape[0],
            "fps": fps
        }

        return stats

    def save_state(self, path: str):
        """Save simulation state to disk"""
        torch.save({
            "frame": self.frame_count,
            "x_mpm": self.x_mpm,
            "v_mpm": self.v_mpm,
            "F": self.F,
            "C": self.C,
            "c_vol": self.c_vol,
            "gaussian_xyz": self.gaussians._xyz,
            "gaussian_opacity": self.gaussians._opacity,
            "gaussian_features_dc": self.gaussians._features_dc
        }, path)
        print(f"[HybridSimulator] State saved to {path}")

    def load_state(self, path: str):
        """Load simulation state from disk"""
        checkpoint = torch.load(path)
        self.frame_count = checkpoint["frame"]
        self.x_mpm = checkpoint["x_mpm"]
        self.v_mpm = checkpoint["v_mpm"]
        self.F = checkpoint["F"]
        self.C = checkpoint["C"]
        self.c_vol = checkpoint["c_vol"]
        self.gaussians._xyz.data = checkpoint["gaussian_xyz"]
        self.gaussians._opacity.data = checkpoint["gaussian_opacity"]
        self.gaussians._features_dc.data = checkpoint["gaussian_features_dc"]
        print(f"[HybridSimulator] State loaded from {path} (frame {self.frame_count})")

    def __repr__(self) -> str:
        return (f"HybridCrackSimulator(particles={self.x_mpm.shape[0] if self.x_mpm is not None else 'N/A'}, "
                f"surface={self.surface_mask.sum().item()}, "
                f"frame={self.frame_count})")


def test_hybrid_simulator():
    """
    Simple test of hybrid simulator integration

    Note: Requires full setup (MPM, Gaussians, elasticity, etc.)
    """
    print("Testing HybridCrackSimulator...")
    print("[Note] This test requires full MPM + GaussianModel setup")
    print("       Run examples/crack_simulation_demo.py for full test\n")

    # Mock test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = 1000
    N_surf = 300

    # Create mock components
    print("Creating mock components...")

    # Mock init positions
    init_pos = torch.rand(N, 3, device=device)  # [0, 1]³

    # Mock surface mask
    surface_mask = torch.zeros(N, dtype=torch.bool, device=device)
    surface_mask[:N_surf] = True

    # Mock coordinate mapper
    mapper = CoordinateMapper(world_scale=2.0, device=device)

    # Mock damage mapper
    damage_mapper = VolumetricToSurfaceDamageMapper(device=device)

    # Mock visualizer
    visualizer = GaussianCrackVisualizer(device=device)

    print(f"✓ Mock components created")
    print(f"  - Particles: {N} ({N_surf} surface)")
    print(f"  - Device: {device}")
    print("\n[Info] Full integration test requires complete setup")
    print("       See examples/crack_simulation_demo.py\n")


if __name__ == "__main__":
    test_hybrid_simulator()
