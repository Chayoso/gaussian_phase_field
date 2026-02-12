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
import math
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

    @torch.no_grad()
    def apply_pre_notch(self, notches: list):
        """
        Seed initial crack from pre-defined notch lines (progressive growth).

        Strategy: Create SHORT seed paths from notch center (2 paths per notch,
        one in each direction). Seed damage along FULL notch line to weaken
        material, but only visualize the short seed. Tips then propagate outward
        over time, creating progressive crack growth.

        Args:
            notches: list of dicts with 'start', 'end', 'damage' keys
        """
        device = self.x_mpm.device
        n = self.mpm.num_grids
        dx = self.mpm.dx
        crack_width = self.pf_params.get('crack_width', 0.03)

        # Ensure grid is initialized
        if not hasattr(self, 'c_grid'):
            self._init_grid_infrastructure()

        # Initialize crack paths
        if not hasattr(self, 'crack_paths'):
            self.crack_paths = []
            self.crack_dirs = []

        H_ref = getattr(self.elasticity, 'Gc', 30.0) / (2.0 * getattr(self.elasticity, 'l0', 0.025))

        for notch in notches:
            start = torch.tensor(notch['start'], device=device, dtype=torch.float32)
            end = torch.tensor(notch['end'], device=device, dtype=torch.float32)
            damage = notch.get('damage', 0.9)

            center = (start + end) * 0.5
            direction = (end - start)
            length = direction.norm().item()
            direction = direction / (direction.norm() + 1e-8)

            # --- 1. Full notch line: sample for damage seeding ---
            n_pts_full = max(2, int(length / dx) + 1)
            t_full = torch.linspace(0.0, 1.0, n_pts_full, device=device)
            full_path = start.unsqueeze(0) + t_full.unsqueeze(1) * (end - start).unsqueeze(0)

            # Seed damage along FULL notch (material weakening, not visual yet)
            min_dist = self._point_to_polyline_dist(self.x_mpm, full_path)
            c_notch = (1.0 - (min_dist / crack_width)).clamp(0.0, 1.0) * damage
            self.c_vol = torch.maximum(self.c_vol, c_notch)

            # Seed H on PARTICLES along notch (so H_grid persists after recompute)
            # _history_H is the per-particle energy history that H_grid is built from
            if not hasattr(self, '_history_H'):
                self._history_H = torch.zeros(self.x_mpm.shape[0], device=device)
            # Particles near notch get high H (guides crack tip propagation)
            H_seed = H_ref * 3.0
            notch_influence = (1.0 - (min_dist / (crack_width * 2.0))).clamp(0.0, 1.0)
            self._history_H = torch.maximum(self._history_H, notch_influence * H_seed)

            # Also seed H_grid directly for frame 0
            if hasattr(self, 'H_grid'):
                self.H_grid = self._bin_particles_to_grid(self._history_H)

            # --- 2. SHORT seed paths from center (2 paths, opposite directions) ---
            seed_len = 3 * dx  # 3 grid cells each direction
            n_seed = max(2, int(seed_len / dx) + 1)

            # Forward path: center → toward end
            t_fwd = torch.linspace(0.0, 1.0, n_seed, device=device)
            path_fwd = center.unsqueeze(0) + t_fwd.unsqueeze(1) * (seed_len * direction).unsqueeze(0)
            self.crack_paths.append(path_fwd)
            self.crack_dirs.append(direction.clone())

            # Backward path: center → toward start
            path_bwd = center.unsqueeze(0) + t_fwd.unsqueeze(1) * (seed_len * (-direction)).unsqueeze(0)
            self.crack_paths.append(path_bwd)
            self.crack_dirs.append(-direction.clone())

            n_damaged = (c_notch > 0.01).sum().item()
            print(f"  [Pre-notch] {notch['start']} → {notch['end']}")
            print(f"    center=[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}]")
            print(f"    seed: 2 paths x {n_seed}pts, damaged particles: {n_damaged}")
            print(f"    material weakened along full {n_pts_full}-point line")

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
    def _compute_aniso_diffusion(self) -> Tensor:
        """
        Compute anisotropic diffusion tensor from stress field.

        Crack opens along max principal stress direction n₁.
        Crack propagates perpendicular to n₁.

        D_tensor = D_min * (n₁⊗n₁) + D_max * (I - n₁⊗n₁)
                 = D_max * I + (D_min - D_max) * (n₁⊗n₁)

        Where:
            D_max = D_base (along crack front, same as isotropic)
            D_min = D_base / aniso_ratio (across crack, much smaller)

        Returns:
            (n, n, n, 6) tensor: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
        """
        n = self.mpm.num_grids
        device = self.x_mpm.device

        D_base = self.pf_params.get('crack_diff_coeff', 0.0005)
        aniso_ratio = self.pf_params.get('crack_aniso_ratio', 10.0)

        D_max = D_base                  # Along crack front (propagation speed unchanged)
        D_min = D_base / aniso_ratio    # Across crack (opening direction, much smaller)

        # Default: isotropic D_base at all cells
        D_tensor = torch.zeros(n, n, n, 6, device=device)
        D_tensor[:, :, :, 0] = D_base  # Dxx
        D_tensor[:, :, :, 1] = D_base  # Dyy
        D_tensor[:, :, :, 2] = D_base  # Dzz
        # Off-diagonal = 0

        if not hasattr(self, '_last_stress'):
            return D_tensor

        # --- 1) Bin 6 stress tensor components to grid ---
        stress = self._last_stress  # (N, 3, 3)
        components = [
            (0, 0), (1, 1), (2, 2),  # diagonal
            (0, 1), (0, 2), (1, 2),  # upper triangle
        ]
        S_comps = []
        for (i, j) in components:
            S_comps.append(self._bin_particles_to_grid(stress[:, i, j]))
        # S_comps: list of 6 (n,n,n) tensors

        # --- 2) Reconstruct symmetric 3x3 at occupied cells ---
        occ = self.grid_occupied
        occ_idx = occ.nonzero()  # (M, 3)
        M = occ_idx.shape[0]
        if M == 0:
            return D_tensor

        S_occ = torch.zeros(M, 3, 3, device=device)
        for k, (i, j) in enumerate(components):
            vals = S_comps[k][occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]]
            S_occ[:, i, j] = vals
            if i != j:
                S_occ[:, j, i] = vals

        # --- 3) Filter cells with significant stress, sanitize ---
        S_occ = torch.nan_to_num(S_occ, 0.0, 0.0, 0.0)
        S_norm = S_occ.norm(dim=(1, 2))
        sig_mask = (S_norm > 1e-3) & torch.isfinite(S_norm)
        if sig_mask.sum() == 0:
            return D_tensor

        sig_idx = occ_idx[sig_mask]  # (K, 3)
        S_sig = S_occ[sig_mask]      # (K, 3, 3)

        # Ensure perfect symmetry (avoid cusolver errors)
        S_sig = 0.5 * (S_sig + S_sig.transpose(1, 2))

        # --- 4) Eigendecomposition → max principal direction ---
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(S_sig)  # sorted ascending
        except Exception:
            return D_tensor
        # Max principal stress = last eigenvalue, direction = last eigenvector
        n1 = eigenvectors[:, :, -1]  # (K, 3) max principal direction

        # --- 4b) Store n1 on full grid for directional reaction modulation ---
        if not hasattr(self, '_n1_grid'):
            self._n1_grid = torch.zeros(n, n, n, 3, device=device)
        self._n1_grid.zero_()
        self._n1_grid[sig_idx[:, 0], sig_idx[:, 1], sig_idx[:, 2]] = n1

        # --- 5) Build D_tensor at these cells ---
        # D = D_max * I + (D_min - D_max) * (n₁⊗n₁)
        dD = D_min - D_max  # negative: reduces D along n₁

        ix = sig_idx[:, 0]
        iy = sig_idx[:, 1]
        iz = sig_idx[:, 2]

        D_tensor[ix, iy, iz, 0] = D_max + dD * n1[:, 0] ** 2  # Dxx
        D_tensor[ix, iy, iz, 1] = D_max + dD * n1[:, 1] ** 2  # Dyy
        D_tensor[ix, iy, iz, 2] = D_max + dD * n1[:, 2] ** 2  # Dzz
        D_tensor[ix, iy, iz, 3] = dD * n1[:, 0] * n1[:, 1]    # Dxy
        D_tensor[ix, iy, iz, 4] = dD * n1[:, 0] * n1[:, 2]    # Dxz
        D_tensor[ix, iy, iz, 5] = dD * n1[:, 1] * n1[:, 2]    # Dyz

        return D_tensor

    @torch.no_grad()
    def _anisotropic_laplacian(self, c: Tensor, D_tensor: Tensor) -> Tensor:
        """
        Compute anisotropic Laplacian: L = Σ_ij D_ij * ∂²c/∂xi∂xj

        Uses central finite differences for all 6 second derivatives.
        Cross terms: ∂²c/∂xi∂xj = (c[+i,+j] - c[+i,-j] - c[-i,+j] + c[-i,-j]) / (4 dx²)

        Args:
            c: (n, n, n) scalar field
            D_tensor: (n, n, n, 6) [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]

        Returns:
            (n, n, n) anisotropic Laplacian
        """
        dx = self.mpm.dx
        dx2 = dx * dx

        # Pad for boundary
        g5 = c.unsqueeze(0).unsqueeze(0)
        gp = torch.nn.functional.pad(g5, (1, 1, 1, 1, 1, 1), mode='replicate')[0, 0]

        # Direct second derivatives (diagonal of Hessian)
        c_xx = (gp[2:, 1:-1, 1:-1] - 2 * c + gp[:-2, 1:-1, 1:-1]) / dx2
        c_yy = (gp[1:-1, 2:, 1:-1] - 2 * c + gp[1:-1, :-2, 1:-1]) / dx2
        c_zz = (gp[1:-1, 1:-1, 2:] - 2 * c + gp[1:-1, 1:-1, :-2]) / dx2

        # Cross second derivatives (off-diagonal of Hessian)
        c_xy = (gp[2:, 2:, 1:-1] - gp[2:, :-2, 1:-1]
                - gp[:-2, 2:, 1:-1] + gp[:-2, :-2, 1:-1]) / (4 * dx2)
        c_xz = (gp[2:, 1:-1, 2:] - gp[2:, 1:-1, :-2]
                - gp[:-2, 1:-1, 2:] + gp[:-2, 1:-1, :-2]) / (4 * dx2)
        c_yz = (gp[1:-1, 2:, 2:] - gp[1:-1, 2:, :-2]
                - gp[1:-1, :-2, 2:] + gp[1:-1, :-2, :-2]) / (4 * dx2)

        # L = Dxx*c_xx + Dyy*c_yy + Dzz*c_zz + 2*Dxy*c_xy + 2*Dxz*c_xz + 2*Dyz*c_yz
        L = (D_tensor[:, :, :, 0] * c_xx +
             D_tensor[:, :, :, 1] * c_yy +
             D_tensor[:, :, :, 2] * c_zz +
             2.0 * D_tensor[:, :, :, 3] * c_xy +
             2.0 * D_tensor[:, :, :, 4] * c_xz +
             2.0 * D_tensor[:, :, :, 5] * c_yz)

        return L

    @torch.no_grad()
    def step_hybrid_crack(self, dt: float):
        """
        Hybrid crack propagation: anisotropic Fisher-KPP with stress-directed diffusion.

        Key improvements over isotropic version:
        1. Anisotropic diffusion: D small along max principal stress (crack opening),
           D large perpendicular (crack propagation) → thin crack lines, not blobs
        2. Top-K nucleation: only seed at the K highest-H cells per frame → cleaner paths
        3. H-modulated reaction rate: alpha(x) ∝ H/H_ref → physics-directed

        Args:
            dt: Physics timestep (not directly used; Fisher-KPP has its own dt)
        """
        if not hasattr(self, 'c_grid'):
            self._init_grid_infrastructure()

        if not hasattr(self, '_hybrid_step'):
            self._hybrid_step = 0

        n = self.mpm.num_grids
        dx = self.mpm.dx
        device = self.x_mpm.device

        step = self._hybrid_step

        # --- Parameters ---
        Gc = getattr(self.elasticity, 'Gc', 30.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)
        H_ref = Gc / (2.0 * l0)
        max_nuc = self.pf_params.get('max_nucleation_per_frame', 1)
        nucleation_frac = self.pf_params.get('nucleation_fraction', 0.3)
        min_spacing = self.pf_params.get('nucleation_min_spacing', 8)
        crack_tip_speed = self.pf_params.get('crack_tip_speed', 1.5)
        crack_width = self.pf_params.get('crack_width', 0.025)
        max_total_cracks = self.pf_params.get('max_total_cracks', 5)

        # Initialize crack paths and smoothed directions
        if not hasattr(self, 'crack_paths'):
            self.crack_paths = []
            self.crack_dirs = []  # smoothed propagation direction per path

        # --- 1) Bin particle H to grid ---
        if hasattr(self, '_history_H'):
            self.H_grid = self._bin_particles_to_grid(self._history_H)
        else:
            self.H_grid = torch.zeros(n, n, n, device=device)

        # --- 2) Compute stress eigenvectors for propagation direction ---
        if hasattr(self, '_last_stress') and self._last_stress is not None:
            self._compute_aniso_diffusion()  # computes self._n1_grid

        # --- 3) Nucleation: create new crack tips at highest-H cells ---
        n_new = 0
        if len(self.crack_paths) < max_total_cracks:
            # Build exclusion zone around existing crack paths
            crack_mask = torch.zeros(n, n, n, device=device, dtype=torch.bool)
            for path in self.crack_paths:
                for pt in path:
                    gi = (pt * n).long().clamp(0, n - 1)
                    lo = (gi - min_spacing).clamp(min=0)
                    hi = (gi + min_spacing + 1).clamp(max=n)
                    crack_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True

            # Interior mask: require all 6 face-neighbors to be occupied
            # This prevents nucleation at thin extremities (ears, nose, feet)
            occ = self.grid_occupied
            interior = (occ[1:-1, 1:-1, 1:-1] &
                        occ[2:, 1:-1, 1:-1] & occ[:-2, 1:-1, 1:-1] &
                        occ[1:-1, 2:, 1:-1] & occ[1:-1, :-2, 1:-1] &
                        occ[1:-1, 1:-1, 2:] & occ[1:-1, 1:-1, :-2])
            interior_full = torch.zeros_like(occ)
            interior_full[1:-1, 1:-1, 1:-1] = interior

            candidate_mask = ((self.H_grid > nucleation_frac * H_ref) &
                              interior_full & ~crack_mask)
            n_candidates = candidate_mask.sum().item()

            if n_candidates > 0 and max_nuc > 0:
                H_score = self.H_grid.clone()
                H_score[~candidate_mask] = 0.0
                K = min(max_nuc, n_candidates)
                _, topk_flat = H_score.view(-1).topk(K)

                for flat_idx in topk_flat:
                    fi = flat_idx.item()
                    i0 = fi // (n * n)       # dim 0
                    i1 = (fi % (n * n)) // n  # dim 1
                    i2 = fi % n               # dim 2
                    pos = torch.tensor(
                        [(i0 + 0.5) / n, (i1 + 0.5) / n, (i2 + 0.5) / n],
                        device=device, dtype=torch.float32
                    )
                    self.crack_paths.append(pos.unsqueeze(0))  # (1, 3)
                    self.crack_dirs.append(None)  # no direction yet
                    n_new += 1
                    if self._hybrid_step < 50:
                        print(f"  [NUC] New crack at grid=({i0},{i1},{i2}) "
                              f"pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                              f"H={self.H_grid[i0,i1,i2]:.1f}", flush=True)

        # --- 4) Advance crack tips (with EMA direction smoothing) ---
        ema_alpha = 0.3  # EMA weight for new direction (lower = smoother)
        min_step_dist = 0.3 * dx  # skip if tip barely moved (anti-jitter)

        # Branching parameters
        branch_angle = self.pf_params.get('branch_angle', 35.0)
        branch_min_len = self.pf_params.get('branch_min_length', 6)
        branch_prob = self.pf_params.get('branch_probability', 0.3)
        max_branches = self.pf_params.get('max_branches_per_path', 1)
        if not hasattr(self, '_branch_count'):
            self._branch_count = {}  # path_idx → number of times branched

        pending_branches = []  # (tip_pos, dir1, dir2) to add after loop

        for path_idx in range(len(self.crack_paths)):
            path = self.crack_paths[path_idx]
            tip = path[-1]  # (3,)

            # Grid index of tip
            gi = (tip * n).long().clamp(1, n - 2)
            i, j, k = gi[0].item(), gi[1].item(), gi[2].item()

            H_local = self.H_grid[i, j, k].item()

            # Compute ∇H at tip (central differences)
            grad_H = torch.zeros(3, device=device)
            grad_H[0] = (self.H_grid[min(i+1, n-1), j, k] - self.H_grid[max(i-1, 0), j, k]) / (2 * dx)
            grad_H[1] = (self.H_grid[i, min(j+1, n-1), k] - self.H_grid[i, max(j-1, 0), k]) / (2 * dx)
            grad_H[2] = (self.H_grid[i, j, min(k+1, n-1)] - self.H_grid[i, j, max(k-1, 0)]) / (2 * dx)

            # Project ∇H perpendicular to n1 (crack opening direction)
            raw_dir = grad_H
            if hasattr(self, '_n1_grid') and self._n1_grid is not None:
                n1 = self._n1_grid[i, j, k]  # (3,)
                n1_mag = n1.norm()
                if n1_mag > 1e-6:
                    n1 = n1 / n1_mag
                    raw_dir = grad_H - (grad_H * n1).sum() * n1

            raw_mag = raw_dir.norm()
            if raw_mag > 1e-8:
                raw_dir = raw_dir / raw_mag
            else:
                # Fallback: continue in previous smoothed direction
                if self.crack_dirs[path_idx] is not None:
                    raw_dir = self.crack_dirs[path_idx]
                elif path.shape[0] >= 2:
                    raw_dir = path[-1] - path[-2]
                    raw_mag = raw_dir.norm()
                    if raw_mag < 1e-8:
                        continue
                    raw_dir = raw_dir / raw_mag
                else:
                    continue

            # EMA smoothing of propagation direction
            if self.crack_dirs[path_idx] is None:
                smooth_dir = raw_dir
            else:
                smooth_dir = (1.0 - ema_alpha) * self.crack_dirs[path_idx] + ema_alpha * raw_dir
                sm = smooth_dir.norm()
                if sm < 1e-8:
                    smooth_dir = raw_dir
                else:
                    smooth_dir = smooth_dir / sm
            self.crack_dirs[path_idx] = smooth_dir

            # Energy-proportional tip speed (Griffith criterion: G ∝ H)
            # Floor at 0.5 so tips keep moving even in low-H regions
            speed_scale = min(H_local / (H_ref + 1e-12), 5.0)
            speed = crack_tip_speed * dx * max(speed_scale, 0.5)
            new_tip = (tip + speed * smooth_dir).clamp(dx, 1.0 - dx)

            # Constrain tip to stay inside occupied grid cells
            ngi = (new_tip * n).long().clamp(0, n - 1)
            if not self.grid_occupied[ngi[0], ngi[1], ngi[2]]:
                # Try 26 neighbors to find an occupied cell nearby
                found = False
                best_tip = None
                best_dot = -1.0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            alt_dir = torch.tensor(
                                [float(di), float(dj), float(dk)], device=device
                            )
                            alt_dir = alt_dir / alt_dir.norm()
                            alt_tip = (tip + speed * alt_dir).clamp(dx, 1.0 - dx)
                            agi = (alt_tip * n).long().clamp(0, n - 1)
                            if self.grid_occupied[agi[0], agi[1], agi[2]]:
                                dot = (alt_dir * smooth_dir).sum()
                                if dot > best_dot:
                                    best_dot = dot
                                    best_tip = alt_tip
                                    found = True
                if found:
                    new_tip = best_tip
                    self.crack_dirs[path_idx] = ((new_tip - tip) /
                                                  ((new_tip - tip).norm() + 1e-8))
                else:
                    continue  # completely stuck, skip

            # Anti-jitter: only extend path if tip actually moved significantly
            if (new_tip - tip).norm() > min_step_dist:
                self.crack_paths[path_idx] = torch.cat([path, new_tip.unsqueeze(0)], dim=0)

            # --- Branching check ---
            n_branches_so_far = self._branch_count.get(path_idx, 0)
            path_len = self.crack_paths[path_idx].shape[0]
            can_branch = (path_len >= branch_min_len and
                          n_branches_so_far < max_branches and
                          len(self.crack_paths) + len(pending_branches) * 2 < max_total_cracks and
                          H_local > 0.3 * H_ref)
            if can_branch and torch.rand(1).item() < branch_prob:
                parent_dir = self.crack_dirs[path_idx]
                if parent_dir is not None:
                    dir1 = self._rotate_direction(parent_dir, branch_angle)
                    dir2 = self._rotate_direction(parent_dir, -branch_angle)
                    pending_branches.append((new_tip.clone(), dir1, dir2))
                    self._branch_count[path_idx] = n_branches_so_far + 1

        # Add branched paths
        for tip_pos, dir1, dir2 in pending_branches:
            idx1 = len(self.crack_paths)
            self.crack_paths.append(tip_pos.unsqueeze(0))
            self.crack_dirs.append(dir1)
            idx2 = len(self.crack_paths)
            self.crack_paths.append(tip_pos.unsqueeze(0))
            self.crack_dirs.append(dir2)
            if step < 50:
                print(f"  [BRANCH] New fork at [{tip_pos[0]:.3f},{tip_pos[1]:.3f},{tip_pos[2]:.3f}] "
                      f"angle=±{branch_angle}°", flush=True)

        # --- 5) AT2 phase field PDE solve (variational damage) ---
        # Seed c_grid from particle damage on first step (pre-notch initial condition)
        if step == 0 and self.c_vol.max() > 0:
            c_from_vol = self._bin_particles_to_grid(self.c_vol)
            self.c_grid = torch.maximum(self.c_grid, c_from_vol)

        # Seed H along tracked crack paths (crack-surface stress singularity)
        # Physical basis: LEFM K → ∞ at crack surface → AT2 limit H → ∞
        # This provides the boundary condition that AT2 needs to create
        # high damage (c → 1) along existing crack paths
        H_crack = H_ref * 5.0
        for path in self.crack_paths:
            gi = (path * n).long().clamp(0, n - 1)  # (M, 3)
            current_H = self.H_grid[gi[:, 0], gi[:, 1], gi[:, 2]]
            self.H_grid[gi[:, 0], gi[:, 1], gi[:, 2]] = torch.maximum(
                current_H, torch.full_like(current_H, H_crack)
            )

        c_old = self.c_vol.clone()
        at2_iters = 50 if step < 3 else 30  # more iterations for cold start
        self._solve_at2_phase_field(n_iters=at2_iters)
        self._gather_grid_to_particles()
        self.c_vol = torch.maximum(self.c_vol, c_old)  # irreversibility

        # --- 6) Diagnostics ---
        if step < 5 or step % 20 == 0:
            n_paths = len(self.crack_paths)
            total_pts = sum(p.shape[0] for p in self.crack_paths)
            max_len = max((p.shape[0] for p in self.crack_paths), default=0)
            n_cracked = (self.c_vol > 0.3).sum().item()
            dc = self.c_vol - c_old
            cg_max = self.c_grid.max().item()
            cg_cells = (self.c_grid > 0.3).sum().item()
            print(f"  [AT2 {step:3d}] paths={n_paths} pts={total_pts} "
                  f"max_seg={max_len} c_vol={self.c_vol.max():.4f} "
                  f"c_grid={cg_max:.4f}({cg_cells}cells) "
                  f"cracked(>0.3)={n_cracked} dc_max={dc.max():.6f} "
                  f"H_max={self.H_grid.max():.2e} H_ref={H_ref:.2e} "
                  f"nuc_new={n_new}",
                  flush=True)

        self._hybrid_step += 1

    @torch.no_grad()
    def _solve_at2_phase_field(self, n_iters: int = 30):
        """
        Solve AT2 phase field equilibrium on grid via Jacobi iteration.

        Minimizes the Ambrosio-Tortorelli (AT2) energy functional:
            E(c) = ∫ [g(c)·H + Gc/(2l₀)(c² + l₀²|∇c|²)] dx

        where g(c) = (1-c)² (quadratic degradation),
              H = max_t ψ(ε(t)) (strain energy history).

        Euler-Lagrange stationarity condition:
            (2H + Gc/l₀)·c - Gc·l₀·Δc = 2H

        Jacobi update:
            c = (2H + (Gc·l₀/dx²)·Σ_nbr c) / (2H + Gc/l₀ + 6·Gc·l₀/dx²)

        Guarantees:
        - Irreversibility: c_new ≥ c_old (damage only grows)
        - Volume constraint: c = 0 outside occupied grid cells
        - Bounds: c ∈ [0, 1]

        Args:
            n_iters: Number of Jacobi iterations (30 typical, 50 for cold start)
        """
        n = self.mpm.num_grids
        dx = self.mpm.dx

        Gc = getattr(self.elasticity, 'Gc', 30.0)
        l0 = getattr(self.elasticity, 'l0', 0.025)

        # Precompute constant coefficients
        Gc_over_l0 = Gc / l0                    # e.g. 1200.0
        Gc_l0_over_dx2 = Gc * l0 / (dx * dx)   # e.g. ~3072 for 64³ grid

        # 2H driving force — (n, n, n)
        H2 = 2.0 * self.H_grid

        # Diagonal: a_ii = 2H + Gc/l₀ + 6·Gc·l₀/dx²
        diag = H2 + Gc_over_l0 + 6.0 * Gc_l0_over_dx2

        c_old = self.c_grid.clone()
        c = self.c_grid.clone()
        occ = self.grid_occupied.float()

        for _ in range(n_iters):
            # Neumann BC via replicate padding
            cp = torch.nn.functional.pad(
                c.unsqueeze(0).unsqueeze(0),
                (1, 1, 1, 1, 1, 1), mode='replicate'
            )[0, 0]

            # Sum of 6 face-adjacent neighbors
            nbr_sum = (cp[2:, 1:-1, 1:-1] + cp[:-2, 1:-1, 1:-1] +
                       cp[1:-1, 2:, 1:-1] + cp[1:-1, :-2, 1:-1] +
                       cp[1:-1, 1:-1, 2:] + cp[1:-1, 1:-1, :-2])

            # Jacobi update: c = (rhs + off-diag) / diag
            c = (H2 + Gc_l0_over_dx2 * nbr_sum) / (diag + 1e-12)
            c = c.clamp(0.0, 1.0)
            c = c * occ           # zero outside object volume
            c = torch.maximum(c, c_old)  # irreversibility

        self.c_grid = c

    @torch.no_grad()
    def _assign_crack_damage(self, crack_width: float = 0.025):
        """Assign damage to particles based on proximity to crack paths (legacy geometric)."""
        if not self.crack_paths:
            return

        positions = self.x_mpm  # (N, 3)
        min_dist = torch.full((positions.shape[0],), float('inf'), device=positions.device)

        for path in self.crack_paths:
            if path.shape[0] == 1:
                dist = (positions - path[0]).norm(dim=1)
            else:
                dist = self._point_to_polyline_dist(positions, path)
            min_dist = torch.minimum(min_dist, dist)

        # Smooth damage profile: 1 at crack center, 0 at crack_width
        c_new = (1.0 - (min_dist / crack_width)).clamp(0.0, 1.0)
        self.c_vol = torch.maximum(self.c_vol, c_new)

    @torch.no_grad()
    def _point_to_polyline_dist(self, points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
        """Compute minimum distance from each point to any segment of a polyline."""
        N = points.shape[0]
        M = polyline.shape[0]

        if M < 2:
            return (points - polyline[0]).norm(dim=1)

        min_dist = torch.full((N,), float('inf'), device=points.device)

        a = polyline[:-1]    # (M-1, 3) segment starts
        b = polyline[1:]     # (M-1, 3) segment ends
        ab = b - a           # (M-1, 3)
        ab_len = ab.norm(dim=1)  # (M-1,)
        valid = ab_len > 1e-10

        if not valid.any():
            return (points - polyline[0]).norm(dim=1)

        a_v = a[valid]
        ab_v = ab[valid]
        ab_len_v = ab_len[valid]
        ab_norm_v = ab_v / ab_len_v.unsqueeze(1)

        for seg_idx in range(a_v.shape[0]):
            ap = points - a_v[seg_idx]                           # (N, 3)
            t = (ap * ab_norm_v[seg_idx]).sum(dim=1)             # (N,)
            t = t.clamp(0.0, ab_len_v[seg_idx].item())
            closest = a_v[seg_idx] + t.unsqueeze(1) * ab_norm_v[seg_idx]  # (N, 3)
            dist = (points - closest).norm(dim=1)
            min_dist = torch.minimum(min_dist, dist)

        return min_dist

    @torch.no_grad()
    def _rotate_direction(self, direction: Tensor, angle_deg: float) -> Tensor:
        """
        Rotate a 3D direction vector by angle_deg around a perpendicular axis.
        Uses Rodrigues' rotation formula.
        """
        device = direction.device
        # Find a perpendicular axis
        if abs(direction[1].item()) < 0.9:
            up = torch.tensor([0.0, 1.0, 0.0], device=device)
        else:
            up = torch.tensor([1.0, 0.0, 0.0], device=device)
        axis = torch.linalg.cross(direction, up)
        axis = axis / (axis.norm() + 1e-8)

        angle = angle_deg * math.pi / 180.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Rodrigues: v' = v*cos + (k x v)*sin + k*(k.v)*(1-cos)
        rotated = (direction * cos_a +
                   torch.linalg.cross(axis, direction) * sin_a +
                   axis * (axis @ direction) * (1.0 - cos_a))
        return rotated / (rotated.norm() + 1e-8)

    @torch.no_grad()
    def _project_paths_to_surface(
        self, crack_paths: list, x_surf_world: Tensor
    ) -> list:
        """
        Project interior crack paths onto the nearest surface Gaussian positions.

        Uses INITIAL (undeformed) surface positions for stable mapping, then
        returns CURRENT positions of those same Gaussians. This prevents crack
        lines from wobbling/jumping between Gaussians during seismic shaking.

        Args:
            crack_paths: list of (M_i, 3) tensors in MPM space
            x_surf_world: (N_surf, 3) CURRENT surface Gaussian positions in world space

        Returns:
            list of (M_i, 3) projected paths in world space (current positions)
        """
        # Cache initial surface positions for stable projection
        if not hasattr(self, '_init_surf_world'):
            self._init_surf_world = x_surf_world.clone()

        # Cache per-path projection indices (only recompute when path grows)
        if not hasattr(self, '_proj_cache'):
            self._proj_cache = {}  # path_id → (prev_len, gaussian_indices, unique_mask)

        projected = []
        for path_idx, path in enumerate(crack_paths):
            if path.shape[0] < 2:
                continue

            path_world = self.mapper.mpm_to_world(path)  # (M, 3)
            M = path_world.shape[0]

            # Check cache: reuse if path hasn't grown
            cache = self._proj_cache.get(path_idx)
            if cache is not None and cache[0] == M:
                gauss_idx, unique_mask = cache[1], cache[2]
            else:
                # Project against INITIAL (stable) surface positions
                gauss_idx = torch.zeros(M, dtype=torch.long, device=path_world.device)
                chunk = 64
                for i in range(0, M, chunk):
                    j = min(i + chunk, M)
                    dists = torch.cdist(path_world[i:j], self._init_surf_world)
                    gauss_idx[i:j] = dists.argmin(dim=1)

                # Build unique mask (remove consecutive duplicates)
                unique_mask = torch.ones(M, dtype=torch.bool, device=path_world.device)
                for i in range(1, M):
                    if gauss_idx[i] == gauss_idx[i - 1]:
                        unique_mask[i] = False

                self._proj_cache[path_idx] = (M, gauss_idx, unique_mask)

            # Use CURRENT positions of the mapped Gaussians
            nearest_pts = x_surf_world[gauss_idx]
            proj_path = nearest_pts[unique_mask]

            if proj_path.shape[0] >= 2:
                projected.append(proj_path)

        return projected if projected else None

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

        # Store stress tensor for anisotropic crack diffusion
        self._last_stress = stress.detach()

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
        # Project crack paths onto surface for accurate visualization
        crack_paths_surface = None
        if hasattr(self, 'crack_paths') and self.crack_paths:
            crack_paths_surface = self._project_paths_to_surface(
                self.crack_paths, x_surf_world
            )
        crack_width = self.mapper.scale_mpm_to_world(
            self.pf_params.get('crack_width', 0.03)
        )
        self.visualizer.update_gaussians(
            self.gaussians, c_surf, x_surf_world,
            preserve_original=True,
            crack_paths=crack_paths_surface,
            crack_width=crack_width
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
