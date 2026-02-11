"""
Main Entry Point for MPM + Gaussian Splatting Crack Simulation

This script orchestrates the entire simulation pipeline:
1. Load configuration from YAML
2. Convert 3D mesh to point clouds
3. Initialize MPM physics with Phase Field
4. Initialize Gaussian Splats for rendering
5. Run simulation with external forces
6. Generate output video

Usage:
    python run.py --config configs/simulation_config.yaml
    python run.py --config my_config.yaml --mesh assets/meshes/bunny.obj
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Optional
import time
import csv

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gaussian-splatting"))

# Project imports
from src.preprocessing.mesh_converter import MeshToPointCloudConverter
from src.core.coordinate_mapper import CoordinateMapper
from src.core.hybrid_simulator import HybridCrackSimulator
from src.constitutive_models.damage_mapper import VolumetricToSurfaceDamageMapper
from src.visualization.gaussian_updater import GaussianCrackVisualizer
from src.mpm_core.mpm_model import MPMModel
from src.constitutive_models.physical_constitutive_models import (
    PhaseFieldElasticity,
    CorotatedPhaseFieldElasticity
)

# Gaussian Splatting imports
try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from scene.cameras import Camera, MiniCam
    from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix
    from utils.sh_utils import SH2RGB
    from torchvision.utils import save_image
except ImportError as e:
    print(f"[Warning] Gaussian Splatting modules not fully available: {e}")
    GaussianModel = None


# ============================================================================
# Configuration & Setup
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MPM Crack Simulation with Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --config configs/simulation_config.yaml
  python run.py --config my_config.yaml --mesh assets/meshes/bunny.obj
  python run.py --config configs/simulation_config.yaml --frames 500
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/simulation_config.yaml",
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--mesh", "-m",
        type=str,
        default=None,
        help="Override mesh path from config"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override output video path from config"
    )

    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=None,
        help="Override total frames from config"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation"
    )

    return parser.parse_args()


def load_config(config_path: str) -> OmegaConf:
    """
    Load and validate YAML configuration

    Args:
        config_path: Path to YAML config file

    Returns:
        OmegaConf configuration object
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Validate required fields
    required_sections = [
        "simulation", "mesh", "particles", "mpm",
        "material", "phase_field", "gaussian_splatting", "rendering"
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    print(f"[Config] Loaded configuration from: {config_path}")
    print(f"  - Simulation: {config.simulation.name}")
    print(f"  - Mesh: {config.mesh.path}")
    print(f"  - Particles: {config.particles.target_count}")
    print(f"  - Frames: {config.rendering.total_frames}")

    return config


def apply_cli_overrides(config: OmegaConf, args) -> OmegaConf:
    """
    Apply command-line argument overrides to config

    Args:
        config: Base configuration
        args: Parsed command-line arguments

    Returns:
        Modified configuration
    """
    if args.mesh is not None:
        config.mesh.path = args.mesh
        print(f"[Config] Override mesh: {args.mesh}")

    if args.output is not None:
        config.output.video_path = args.output
        print(f"[Config] Override output: {args.output}")

    if args.frames is not None:
        config.rendering.total_frames = args.frames
        print(f"[Config] Override frames: {args.frames}")

    if args.device is not None:
        config.device.type = args.device
        print(f"[Config] Override device: {args.device}")

    return config


# ============================================================================
# Mesh & Point Cloud Processing
# ============================================================================

def setup_mesh(config: OmegaConf):
    """
    Load or generate mesh, convert to point clouds

    Args:
        config: Simulation configuration

    Returns:
        Tuple of (volume_pcd, surface_pcd, surface_mask)
    """
    print(f"\n{'='*60}")
    print(f"Step 1: Mesh Processing")
    print(f"{'='*60}")

    mesh_path = Path(config.mesh.path)

    # Check if mesh exists, generate if needed
    if not mesh_path.exists() and config.mesh.auto_generate_if_missing:
        print(f"[Mesh] File not found: {mesh_path}")
        print(f"[Mesh] Generating test mesh: {config.mesh.test_mesh_type}")

        import open3d as o3d

        if config.mesh.test_mesh_type == "sphere":
            mesh = o3d.geometry.TriangleMesh.create_sphere(
                radius=config.mesh.test_mesh_radius
            )
        elif config.mesh.test_mesh_type == "cube":
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=config.mesh.test_mesh_radius * 2,
                height=config.mesh.test_mesh_radius * 2,
                depth=config.mesh.test_mesh_radius * 2
            )
        elif config.mesh.test_mesh_type == "torus":
            mesh = o3d.geometry.TriangleMesh.create_torus(
                torus_radius=config.mesh.test_mesh_radius,
                tube_radius=config.mesh.test_mesh_radius * 0.3
            )
        else:
            raise ValueError(f"Unknown test mesh type: {config.mesh.test_mesh_type}")

        mesh.compute_vertex_normals()

        # Save to assets
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"[Mesh] Generated and saved: {mesh_path}")

    # Convert mesh to point clouds
    converter = MeshToPointCloudConverter(
        mesh_path=str(mesh_path),
        target_particle_count=config.particles.target_count,
        surface_sample_ratio=config.particles.surface_ratio,
        use_poisson=config.particles.use_poisson_sampling,
        poisson_depth=config.particles.poisson_depth,
        normalize_to_unit_cube=config.particles.normalize_to_unit_cube
    )

    volume_pcd, surface_pcd, surface_mask = converter.convert()

    return volume_pcd, surface_pcd, surface_mask


# ============================================================================
# Simulation Components Initialization
# ============================================================================

def setup_mpm(config: OmegaConf, volume_pcd, device: torch.device):
    """
    Initialize MPM simulation model

    Args:
        config: Simulation configuration
        volume_pcd: Volumetric point cloud
        device: PyTorch device

    Returns:
        MPMModel instance
    """
    print(f"\n{'='*60}")
    print(f"Step 2: MPM Initialization")
    print(f"{'='*60}")

    # Create MPM configuration
    sim_params = OmegaConf.create({
        "num_grids": config.mpm.num_grids,
        "dt": config.mpm.dt,
        "gravity": config.mpm.gravity,
        "clip_bound": config.mpm.clip_bound,
        "damping": config.mpm.damping
    })

    material_params = OmegaConf.create({
        "center": config.material.center,
        "size": config.material.size,
        "rho": config.material.density
    })

    mpm_model = MPMModel(
        sim_params=sim_params,
        material_params=material_params,
        init_pos=torch.from_numpy(volume_pcd.points).float().to(device),
        device=device
    )

    # Set chunk size if specified
    if hasattr(config.mpm, 'particle_chunk'):
        mpm_model.particle_chunk = config.mpm.particle_chunk

    return mpm_model


def setup_elasticity(config: OmegaConf, device: torch.device):
    """
    Initialize elasticity model based on config.

    Args:
        config: Simulation configuration
        device: PyTorch device

    Returns:
        Elasticity module (PhaseFieldElasticity or CorotatedPhaseFieldElasticity)
    """
    print(f"\n{'='*60}")
    print(f"Step 3: Elasticity Model")
    print(f"{'='*60}")

    model_type = config.material.get("constitutive_model", "phase_field")

    Gc = float(config.material.get("Gc", 100.0))
    l0 = float(config.material.get("l0", 0.03))

    if model_type == "corotated_phase_field":
        elasticity = CorotatedPhaseFieldElasticity(Gc=Gc, l0=l0).to(device)
    else:
        elasticity = PhaseFieldElasticity().to(device)
        elasticity.Gc = Gc
        elasticity.l0 = l0

    # Override buffer values with config parameters
    elasticity.log_E = torch.log(torch.tensor([config.material.youngs_modulus], device=device))
    elasticity.nu = torch.tensor([config.material.poissons_ratio], device=device)

    # Set damage degradation exponent (accessed via getattr in forward())
    elasticity.damage_exp = config.material.degradation_exponent

    print(f"  - Model: {model_type}")
    print(f"  - Young's modulus: {config.material.youngs_modulus:.2e}")
    print(f"  - Poisson's ratio: {config.material.poissons_ratio}")
    print(f"  - Gc: {Gc:.1f}, l0: {l0:.4f}")
    print(f"  - Degradation exponent: {config.material.degradation_exponent}")

    return elasticity


def setup_gaussians(config: OmegaConf, surface_pcd, device: torch.device):
    """
    Initialize Gaussian Splatting model

    Args:
        config: Simulation configuration
        surface_pcd: Surface point cloud
        device: PyTorch device

    Returns:
        GaussianModel instance
    """
    print(f"\n{'='*60}")
    print(f"Step 4: Gaussian Splats Initialization")
    print(f"{'='*60}")

    if GaussianModel is None:
        raise ImportError("GaussianModel not available. Install Gaussian Splatting submodules.")

    # Compute camera parameters for scale initialization
    cam_distance = config.rendering.camera.distance
    cam_elevation = np.radians(config.rendering.camera.elevation)
    cam_azimuth = np.radians(config.rendering.camera.azimuth)
    cam_fov = np.radians(config.rendering.camera.fov)
    img_width = config.rendering.image_width
    img_height = config.rendering.image_height
    aspect_ratio = img_width / img_height
    fov_y = cam_fov
    fov_x = 2 * np.arctan(np.tan(fov_y / 2) * aspect_ratio)

    print(f"\n[Camera Parameters for Scale Init]")
    print(f"  - Distance: {cam_distance:.3f}")
    print(f"  - Image: {img_width}x{img_height}")
    print(f"  - FoV: {np.degrees(fov_x):.1f}° x {np.degrees(fov_y):.1f}°")

    gaussians = GaussianModel(sh_degree=config.gaussian_splatting.sh_degree)
    gaussians.create_from_pcd(
        surface_pcd,
        cam_infos=[],
        spatial_lr_scale=1.0,
        camera_distance=cam_distance,
        image_width=img_width,
        fov_x=fov_x
    )

    # Move to device
    gaussians._xyz = gaussians._xyz.to(device)
    gaussians._features_dc = gaussians._features_dc.to(device)
    gaussians._features_rest = gaussians._features_rest.to(device)
    gaussians._opacity = gaussians._opacity.to(device)
    gaussians._scaling = gaussians._scaling.to(device)
    gaussians._rotation = gaussians._rotation.to(device)

    print(f"  - Surface Gaussians: {gaussians._xyz.shape[0]}")
    print(f"  - SH degree: {config.gaussian_splatting.sh_degree}")

    return gaussians


def setup_simulator(
    config: OmegaConf,
    mpm_model,
    gaussians,
    elasticity,
    surface_mask,
    device: torch.device
):
    """
    Create hybrid MPM + Gaussian Splats simulator

    Args:
        config: Simulation configuration
        mpm_model: MPM model
        gaussians: Gaussian model
        elasticity: Elasticity module
        surface_mask: Surface particle mask
        device: PyTorch device

    Returns:
        HybridCrackSimulator instance
    """
    print(f"\n{'='*60}")
    print(f"Step 5: Hybrid Simulator Setup")
    print(f"{'='*60}")

    # Coordinate mapper
    mapper = CoordinateMapper(
        world_center=np.array(config.coordinate_mapping.world_center),
        world_scale=config.coordinate_mapping.world_scale,
        device=device
    )

    # Damage mapper
    damage_mapper = VolumetricToSurfaceDamageMapper(
        projection_method=config.damage_projection.method,
        k_neighbors=config.damage_projection.k_neighbors,
        influence_radius=config.damage_projection.influence_radius,
        damage_threshold=config.damage_projection.damage_threshold,
        use_faiss=config.damage_projection.use_faiss,
        device=device
    )

    # Visualizer
    visualizer = GaussianCrackVisualizer(
        crack_color=tuple(config.gaussian_splatting.crack_color),
        opacity_mode=config.gaussian_splatting.opacity_mode,
        color_mode=config.gaussian_splatting.color_mode,
        scale_mode=config.gaussian_splatting.scale_mode,
        damage_threshold=config.gaussian_splatting.get("damage_threshold", 0.2),
        sharp_k=config.gaussian_splatting.get("sharp_k", 30.0),
        edge_width=config.gaussian_splatting.get("edge_width", 0.12),
        crack_opacity_reduction=config.gaussian_splatting.get("crack_opacity_reduction", 0.85),
        max_opening=config.gaussian_splatting.get("crack_max_opening", 0.015),
        gap_fraction=config.gaussian_splatting.get("crack_gap_fraction", 0.3),
        edge_darken=config.gaussian_splatting.get("crack_edge_darken", 0.3),
        red_accent=config.gaussian_splatting.get("crack_red_accent", 0.15),
        device=device
    )

    # Phase field parameters (AT2 uses Gc/l0 from elasticity, only need warmup + rate limiter)
    phase_field_params = {
        "warmup_frames": config.phase_field.get("warmup_frames", 5),
        "dC_max": config.phase_field.get("dC_max", 0.02),
        # Crack-tip tracking parameters
        "crack_tip_speed": config.phase_field.get("crack_tip_speed", 1.5),
        "crack_width": config.phase_field.get("crack_width", 0.025),
        "max_total_cracks": config.phase_field.get("max_total_cracks", 5),
        # Anisotropic diffusion (for stress eigenvector computation)
        "crack_aniso_ratio": config.phase_field.get("crack_aniso_ratio", 200.0),
        # Nucleation parameters
        "nucleation_fraction": config.phase_field.get("nucleation_fraction", 0.3),
        "max_nucleation_per_frame": config.phase_field.get("max_nucleation_per_frame", 1),
        "nucleation_min_spacing": config.phase_field.get("nucleation_min_spacing", 8),
    }

    # Seismic loading parameters (earthquake ground motion)
    seismic_params = {}
    if hasattr(config, 'seismic'):
        seismic_params = {
            "enabled": config.seismic.get("enabled", False),
            "amplitude": float(config.seismic.get("amplitude", 1000.0)),
            "frequency": float(config.seismic.get("frequency", 80.0)),
            "direction": list(config.seismic.get("direction", [1.0, 0.0, 0.0])),
            "ramp_time": float(config.seismic.get("ramp_time", 0.005)),
        }

    # Create simulator
    simulator = HybridCrackSimulator(
        mpm_model=mpm_model,
        gaussians=gaussians,
        elasticity_module=elasticity,
        coord_mapper=mapper,
        damage_mapper=damage_mapper,
        visualizer=visualizer,
        surface_mask=torch.from_numpy(surface_mask).to(device),
        physics_substeps=config.rendering.physics_substeps,
        phase_field_params=phase_field_params,
        simulation_mode=config.simulation.get("mode", "crack_only"),
        seismic_params=seismic_params
    )

    return simulator


# ============================================================================
# Rendering
# ============================================================================

def setup_camera(config: OmegaConf):
    """
    Create rendering camera using proper camera system.

    Args:
        config: Simulation configuration

    Returns:
        MiniCam object with proper intrinsics and projection
    """
    from src.renderer.camera.config import make_matrices_from_yaml

    # Build camera configuration dictionary for the proper camera system
    cam_config = config.rendering.camera
    width = config.rendering.image_width
    height = config.rendering.image_height

    # Convert orbital parameters (distance, elevation, azimuth) to lookat parameters
    elev_rad = np.radians(cam_config.elevation)
    azim_rad = np.radians(cam_config.azimuth)
    distance = cam_config.distance
    target = np.array([0.5, 0.5, 0.5])  # Mesh center

    # Compute eye position from orbital parameters
    x = target[0] + distance * np.cos(elev_rad) * np.cos(azim_rad)
    y = target[1] + distance * np.cos(elev_rad) * np.sin(azim_rad)
    z = target[2] + distance * np.sin(elev_rad)
    eye = [float(x), float(y), float(z)]

    # Compute intrinsic parameters from FOV
    fov_deg = cam_config.fov
    fov_rad = np.radians(fov_deg)
    # fx = width / (2 * tan(fov_x / 2))
    # For square FOV: use width as reference
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # Assuming square pixels
    cx = width / 2.0
    cy = height / 2.0

    # Build proper camera configuration
    camera_yaml = {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "znear": 0.01,
        "zfar": 100.0,
        "lookat": {
            "eye": eye,
            "target": [0.5, 0.5, 0.5],
            "up": [0.0, 0.0, 1.0]  # Z-up
        }
    }

    print(f"\n[Camera Setup]")
    print(f"  - Eye: [{eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f}]")
    print(f"  - Target: [0.5, 0.5, 0.5]")
    print(f"  - Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"  - FOV: {fov_deg}°")

    # Use proper camera system to build matrices
    w, h, tanfovx, tanfovy, view_matrix, proj_matrix, camera_position = make_matrices_from_yaml(camera_yaml)

    # Convert to PyTorch tensors
    world_view_transform = torch.from_numpy(view_matrix).cuda()
    full_proj_transform = torch.from_numpy(proj_matrix).cuda()

    # Compute FOV angles from tan_half_fov
    fov_x = 2.0 * np.arctan(tanfovx)
    fov_y = 2.0 * np.arctan(tanfovy)

    # Camera position as tensor
    camera_center_tensor = torch.from_numpy(camera_position).cuda()

    # Use MiniCam (simpler, no image data required)
    camera = MiniCam(
        width=w,
        height=h,
        fovy=fov_y,
        fovx=fov_x,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform
    )

    return camera


# ============================================================================
# Simulation Loop
# ============================================================================

def run_simulation(config: OmegaConf, simulator, camera, args):
    """
    Main simulation loop

    Args:
        config: Simulation configuration
        simulator: Hybrid simulator
        camera: Rendering camera
        args: Command-line arguments
    """
    print(f"\n{'='*60}")
    print(f"Step 6: Running Simulation")
    print(f"{'='*60}")

    # Setup output directories
    output_dir = Path(config.output.video_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = Path(config.output.frame_dir)
    if config.simulation.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config.output.checkpoint_dir)
    if config.simulation.save_checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup rendering
    device = torch.device(config.device.type)
    bg_color = torch.tensor(config.rendering.background_color, device=device)

    # Create pipeline parameters for Gaussian Splatting renderer
    pipe = type('obj', (object,), {
        'convert_SHs_python': False,  # Let rasterizer handle SH to RGB conversion
        'compute_cov3D_python': False,
        'debug': True,  # Enable debug mode to see C++ errors
        'antialiasing': False
    })()

    # Apply impact and initialize crack energy field
    if config.external_force.enabled:
        print(f"\n[Impact] Finding front-facing surface particle for crack initiation...")

        # Test render to find visible Gaussians
        test_rendering = render(camera, simulator.gaussians, pipe, bg_color)
        visible_radii = test_rendering['radii']
        visible_mask = visible_radii > 0
        visible_indices = torch.where(visible_mask)[0]

        print(f"  - Total Gaussians: {len(visible_radii)}")
        print(f"  - Visible from camera: {len(visible_indices)}")

        if len(visible_indices) > 0:
            # Find a front-facing particle as impact center
            visible_positions = simulator.gaussians.get_xyz[visible_indices]
            visible_normals = simulator.gaussians.get_normal[visible_indices]

            cam_pos = camera.camera_center
            view_dirs = cam_pos - visible_positions
            view_dirs = view_dirs / (view_dirs.norm(dim=1, keepdim=True) + 1e-8)
            facing_scores = (visible_normals * view_dirs).sum(dim=1)

            front_facing = facing_scores > 0.3
            if front_facing.sum() > 0:
                front_facing_indices = visible_indices[front_facing]
                center_idx = front_facing_indices[torch.randint(0, len(front_facing_indices), (1,))].item()
                print(f"  - Sampled from {len(front_facing_indices)} front-facing particles")
            else:
                center_idx = visible_indices[facing_scores.argmax()].item()
                print(f"  - Picked best facing particle")

            center_pos = simulator.gaussians.get_xyz[center_idx:center_idx+1]
            print(f"  - Impact center index: {center_idx}")
            print(f"  - Impact position (world): {center_pos[0].detach().cpu().numpy()}")

            # Convert to MPM space
            center_mpm = simulator.mapper.world_to_mpm(center_pos)

            # Initialize based on simulation mode
            sim_mode = config.simulation.get("mode", "crack_only")
            if sim_mode == "crack_only":
                simulator.initialize_crack_energy(
                    impact_center_mpm=center_mpm[0],
                    impact_energy=config.external_force.magnitude,
                    impact_radius=config.external_force.radius
                )
            else:
                # Impact direction: camera → object (inward)
                impact_dir = center_pos[0] - cam_pos
                impact_dir = impact_dir / (impact_dir.norm() + 1e-8)
                # Convert direction to MPM space (same rotation, just scale)
                impact_dir_mpm = simulator.mapper.world_to_mpm(
                    center_pos + impact_dir.unsqueeze(0) * 0.01
                ) - center_mpm
                impact_dir_mpm = impact_dir_mpm[0] / (impact_dir_mpm[0].norm() + 1e-8)

                simulator.initialize_deformation_impact(
                    impact_center_mpm=center_mpm[0],
                    impact_energy=config.external_force.magnitude,
                    impact_radius=config.external_force.radius,
                    impact_direction=impact_dir_mpm
                )
        else:
            print(f"  - WARNING: No visible Gaussians found! Skipping impact.")

    # Apply pre-notch if configured
    if hasattr(config, 'pre_notch') and config.pre_notch.get('enabled', False):
        notches = config.pre_notch.get('notches', [])
        if notches:
            print(f"\n[Pre-notch] Seeding {len(notches)} notch(es) in body...")
            notch_list = []
            for n in notches:
                notch_list.append({
                    'start': list(n['start']),
                    'end': list(n['end']),
                    'damage': n.get('damage', 0.9)
                })
            simulator.apply_pre_notch(notch_list)

    # Statistics logging
    stats_log = []
    log_file = Path(config.output.statistics_log)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Simulation loop
    render_interval = 1  # Render every frame for debugging
    print(f"\nSimulating {config.rendering.total_frames} frames (render every {render_interval})...")
    print(f"{'='*60}")

    start_time = time.time()

    for frame in range(config.rendering.total_frames):
        # Physics + Gaussian update (always runs)
        simulator.step_rendering()

        # Determine if we should render this frame
        should_render = (frame % render_interval == 0) or (frame == config.rendering.total_frames - 1)

        # Skip rendering for intermediate frames (physics only)
        if not should_render:
            # Still collect stats and log periodically
            if frame % config.output.log_interval == 0:
                stats = simulator.get_statistics()
                stats["frame"] = frame
                stats_log.append(stats)
                elapsed = time.time() - start_time
                eta = elapsed / (frame + 1) * (config.rendering.total_frames - frame - 1)
                print(f"Frame {frame:04d}/{config.rendering.total_frames}: "
                      f"c_max={stats['c_max']:.4f}, "
                      f"c_mean={stats['c_mean']:.4f}, "
                      f"cracked={stats['n_cracked']}/{stats['n_particles']}, "
                      f"fps={(frame+1)/elapsed:.1f}, "
                      f"ETA={eta/60:.1f}min", flush=True)
            continue

        print(f"\n--- Rendering frame {frame} ---")

        # Debug: Print Gaussian stats on first frame
        if frame == 0:
            print(f"\n[Debug] Gaussian Stats:")
            print(f"  - Positions: {simulator.gaussians.get_xyz.shape}, range [{simulator.gaussians.get_xyz.min().item():.3f}, {simulator.gaussians.get_xyz.max().item():.3f}]")
            print(f"  - Opacity: {simulator.gaussians.get_opacity.shape}, mean {simulator.gaussians.get_opacity.mean().item():.3f}")
            print(f"  - Scales: {simulator.gaussians.get_scaling.shape}, mean {simulator.gaussians.get_scaling.mean().item():.3f}")
            print(f"  - Features DC: {simulator.gaussians.get_features_dc.shape}, mean {simulator.gaussians.get_features_dc.mean().item():.3f}")
            print(f"  - Features Rest: {simulator.gaussians.get_features_rest.shape}, mean {simulator.gaussians.get_features_rest.mean().item():.3f}")
            print(f"\n[Debug] Camera:")
            print(f"  - Position: {camera.camera_center}")
            print(f"  - FoV: {np.degrees(camera.FoVx):.1f}° x {np.degrees(camera.FoVy):.1f}°")
            print(f"  - Image: {camera.image_width}x{camera.image_height}")

            # Calculate mesh bounds and screen projection
            gauss_pos = simulator.gaussians.get_xyz.detach().cpu().numpy()
            mesh_min = gauss_pos.min(axis=0)
            mesh_max = gauss_pos.max(axis=0)
            mesh_center = (mesh_min + mesh_max) / 2
            mesh_extent = mesh_max - mesh_min

            print(f"\n[Debug] Mesh Bounds:")
            print(f"  - Min: [{mesh_min[0]:.3f}, {mesh_min[1]:.3f}, {mesh_min[2]:.3f}]")
            print(f"  - Max: [{mesh_max[0]:.3f}, {mesh_max[1]:.3f}, {mesh_max[2]:.3f}]")
            print(f"  - Center: [{mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f}]")
            print(f"  - Extent: [{mesh_extent[0]:.3f}, {mesh_extent[1]:.3f}, {mesh_extent[2]:.3f}]")

            # Calculate distance from camera to mesh center
            cam_pos = camera.camera_center.detach().cpu().numpy()
            dist_to_mesh = np.linalg.norm(cam_pos - mesh_center)
            print(f"  - Distance from camera to mesh center: {dist_to_mesh:.3f}")

            # Calculate angular size of mesh
            max_extent = mesh_extent.max()
            angular_size_rad = 2 * np.arctan(max_extent / (2 * dist_to_mesh))
            angular_size_deg = np.degrees(angular_size_rad)
            fov_deg = np.degrees(camera.FoVx)
            screen_coverage = (angular_size_deg / fov_deg) * 100

            print(f"  - Mesh max extent: {max_extent:.3f} units")
            print(f"  - Angular size: {angular_size_deg:.1f}° (covers {screen_coverage:.1f}% of FoV)")
            print(f"  - Recommended distance for 80% coverage: {max_extent / (2 * np.tan(np.radians(fov_deg * 0.8 / 2))):.3f}")

            # Debug: Check view space z-coordinates to understand frustum culling
            print(f"\n[Debug] View Space Coordinates (checking p_view.z for frustum culling):")
            viewmatrix = camera.world_view_transform.detach().cpu().numpy()
            gauss_pos = simulator.gaussians.get_xyz.detach().cpu().numpy()
            print(f"  - Viewmatrix shape: {viewmatrix.shape}")
            print(f"  - Viewmatrix:\n{viewmatrix}")

            # Check 10 sample Gaussians
            sample_idx = np.linspace(0, len(gauss_pos)-1, min(10, len(gauss_pos)), dtype=int)
            print(f"\n  - Sample Gaussians (checking if p_view.z <= 0.2 causes culling):")
            for i in sample_idx:
                pos_hom = np.append(gauss_pos[i], 1.0)  # [x, y, z, 1]
                p_view_hom = viewmatrix @ pos_hom  # 4x4 @ 4 = 4
                p_view = p_view_hom[:3]  # [x, y, z] in view space
                culled = "CULLED" if p_view[2] <= 0.2 else "visible"
                print(f"    [{i:5d}] world={gauss_pos[i]}, view.z={p_view[2]:8.3f} ({culled})")

            print(f"\n  - Culling threshold: p_view.z <= 0.2 (see auxiliary.h line 155)")

        # Render (colors already modified by visualizer in step_rendering)
        rendering = render(
            camera,
            simulator.gaussians,
            pipe,
            bg_color
        )
        image = rendering["render"]
        depth = rendering["depth"]
        normal = rendering.get("normal", None)

        # Apply camera-aware lighting using normals
        if normal is not None:
            object_mask = (depth > 0).float()  # (1, H, W)

            cam_pos = camera.camera_center
            scene_center = torch.tensor([0.5, 0.5, 0.5], device='cuda')
            light_dir = scene_center - cam_pos
            light_dir = light_dir / (light_dir.norm() + 1e-8)

            light_dir_reshaped = light_dir.view(3, 1, 1)
            diffuse = (normal * light_dir_reshaped).sum(dim=0, keepdim=True)
            diffuse = torch.clamp(diffuse, 0.0, 1.0)

            ambient = 0.85
            lit_intensity = ambient + (1.0 - ambient) * diffuse

            image_lit = image * lit_intensity
            bg_intensity = 0.92
            image_bg = image * bg_intensity

            image = image_bg * (1.0 - object_mask) + image_lit * object_mask
            image = torch.clamp(image, 0.0, 1.0)

        if frame == 0:
            radii = rendering['radii']
            n_rendered = (radii > 0).sum().item()
            n_total = radii.shape[0]
            render_percent = 100.0 * n_rendered / n_total

            print(f"\n[Debug] Rendering Results:")
            print(f"  - Image shape: {image.shape}")
            print(f"  - Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
            print(f"  - Depth shape: {depth.shape}")
            print(f"  - Depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}]")
            print(f"  - Depth non-zero pixels: {(depth > 0).sum().item()} / {depth.numel()}")
            if normal is not None:
                print(f"  - Normal shape: {normal.shape}")
                print(f"  - Normal range: [{normal.min().item():.3f}, {normal.max().item():.3f}]")
                print(f"  - Normal non-zero pixels: {(normal.abs() > 0.01).sum().item()} / {normal.numel()}")
                # Check mean normal per channel
                print(f"  - Normal mean (XYZ): [{normal[0].mean().item():.3f}, {normal[1].mean().item():.3f}, {normal[2].mean().item():.3f}]")
            else:
                print(f"  - Normal: None (not returned by rasterizer)")
            print(f"\n[Gaussian Rendering Statistics]")
            print(f"  - Total Gaussians: {n_total}")
            print(f"  - Rendered (radii > 0): {n_rendered} ({render_percent:.1f}%)")
            print(f"  - Culled: {n_total - n_rendered} ({100 - render_percent:.1f}%)")
            if n_rendered > 0:
                print(f"  - Radii range (rendered only): {radii[radii > 0].min().item():.1f} to {radii[radii > 0].max().item():.1f} pixels")
                # Show which Gaussians are rendering
                rendered_indices = torch.where(radii > 0)[0]
                print(f"  - Rendered Gaussian indices: {rendered_indices.cpu().numpy()[:10]}")  # Show first 10
                gauss_pos = simulator.gaussians.get_xyz.detach().cpu().numpy()
                for idx in rendered_indices[:5]:  # Show first 5
                    pos = gauss_pos[idx]
                    print(f"    - Gaussian {idx}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], radius={radii[idx].item():.1f}px")

            # Analyze why Gaussians are culled
            print(f"\n[Culling Analysis]")
            gauss_scales_linear = simulator.gaussians.get_scaling.detach().cpu().numpy()  # Already linear (exp applied)
            gauss_scales_log = simulator.gaussians._scaling.detach().cpu().numpy()  # Raw log values
            gauss_opacity = simulator.gaussians.get_opacity.detach().cpu().numpy()
            print(f"  - Scale (linear, world units): min={gauss_scales_linear.min():.6f}, max={gauss_scales_linear.max():.6f}, mean={gauss_scales_linear.mean():.6f}")
            print(f"  - Scale (log space): min={gauss_scales_log.min():.3f}, max={gauss_scales_log.max():.3f}, mean={gauss_scales_log.mean():.3f}")
            print(f"  - Opacity (sigmoid): min={gauss_opacity.min():.3f}, max={gauss_opacity.max():.3f}, mean={gauss_opacity.mean():.3f}")

            # Check if low rendering is due to scale or culling
            if render_percent < 50:
                print(f"\n  [WARNING] Less than 50% of Gaussians are rendering!")
                print(f"  Possible causes:")
                print(f"    1. Scales too small -> increase target_pixel_coverage or camera distance")
                print(f"    2. Frustum culling -> adjust camera position/orientation")
                print(f"    3. Opacity too low -> increase initial opacity")
            elif render_percent > 95:
                print(f"\n  [OK] Good: >95% of Gaussians are rendering")

        # Save frame if requested
        if config.simulation.save_frames:
            save_image(image, frame_dir / f"frame_{frame:04d}.png")

            # Save depth map
            # Normalize depth to [0, 1] for visualization
            depth_vis = depth.squeeze()  # Remove channel dimension if present
            if depth_vis.numel() > 0:
                depth_min = depth_vis.min()
                depth_max = depth_vis.max()
                if depth_max > depth_min:
                    depth_normalized = (depth_vis - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = torch.zeros_like(depth_vis)

                # Convert to 3-channel for saving (grayscale)
                depth_rgb = depth_normalized.unsqueeze(0).repeat(3, 1, 1)
                save_image(depth_rgb, frame_dir / f"depth_{frame:04d}.png")

            # Save normal map
            if normal is not None:
                # Create object mask from depth (background should be black)
                object_mask_normal = (depth > 0).float().repeat(3, 1, 1)  # (3, H, W)

                # Normal map is typically (3, H, W) with values in [-1, 1]
                # Convert to [0, 1] for visualization: (normal + 1) / 2
                normal_vis = (normal + 1.0) / 2.0
                normal_vis = torch.clamp(normal_vis, 0, 1)

                # Apply mask: object pixels = normal, background = black (0, 0, 0)
                normal_vis = normal_vis * object_mask_normal  # Background becomes (0, 0, 0)

                save_image(normal_vis, frame_dir / f"normal_{frame:04d}.png")

        # Save checkpoint if requested
        if config.simulation.save_checkpoint:
            if frame % config.simulation.checkpoint_interval == 0 and frame > 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{frame:04d}.pth"
                simulator.save_state(str(checkpoint_path))

        # Collect statistics
        stats = simulator.get_statistics()
        stats["frame"] = frame
        stats_log.append(stats)

        # Log progress
        if frame % config.output.log_interval == 0 or frame == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (frame + 1) * (config.rendering.total_frames - frame - 1)

            print(f"Frame {frame:04d}/{config.rendering.total_frames}: "
                  f"c_max={stats['c_max']:.4f}, "
                  f"c_mean={stats['c_mean']:.4f}, "
                  f"cracked={stats['n_cracked']}/{stats['n_particles']}, "
                  f"fps={stats['fps']:.1f}, "
                  f"ETA={eta/60:.1f}min")

    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"Simulation complete! Total time: {total_time/60:.1f} min")

    # Save statistics to CSV
    if stats_log:
        with open(log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats_log[0].keys())
            writer.writeheader()
            writer.writerows(stats_log)
        print(f"[Output] Statistics saved: {log_file}")

    # Generate video
    if not args.no_video and config.simulation.save_frames:
        print(f"\n[Output] Generating video...")
        create_video(frame_dir, config.output.video_path, config.rendering.fps)


def create_video(frame_dir: Path, output_path: str, fps: int):
    """
    Create video from saved frames

    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
    """
    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not available, skipping video creation")
        return

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        print(f"[Warning] No frames found in {frame_dir}")
        return

    # Read first frame for dimensions
    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Write frames
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        out.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames...")

    out.release()
    print(f"[Output] Video created: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()

    # Load and apply configuration
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Setup device
    device_type = config.device.type
    if device_type == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device_type = "cpu"

    device = torch.device(device_type)
    if device_type == "cuda" and hasattr(config.device, 'gpu_id'):
        torch.cuda.set_device(config.device.gpu_id)

    print(f"\n{'='*60}")
    print(f"MPM + Gaussian Splatting Crack Simulation")
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    try:
        # Pipeline execution
        volume_pcd, surface_pcd, surface_mask = setup_mesh(config)
        mpm_model = setup_mpm(config, volume_pcd, device)
        elasticity = setup_elasticity(config, device)
        gaussians = setup_gaussians(config, surface_pcd, device)

        simulator = setup_simulator(
            config, mpm_model, gaussians, elasticity,
            surface_mask, device
        )

        # Initialize simulation
        simulator.initialize(torch.from_numpy(volume_pcd.points).float().to(device))

        # Setup camera
        camera = setup_camera(config)

        # Run simulation
        run_simulation(config, simulator, camera, args)

        print(f"\n{'='*60}")
        print(f"Simulation Complete!")
        print(f"{'='*60}")
        print(f"Output: {config.output.video_path}")
        print(f"Frames: {config.output.frame_dir}")
        print(f"Statistics: {config.output.statistics_log}")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n[Info] Simulation interrupted by user")
    except Exception as e:
        print(f"\n[Error] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
