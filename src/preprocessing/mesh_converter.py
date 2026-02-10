"""
Mesh to Point Cloud Converter

Converts 3D meshes (OBJ/PLY) to dual point cloud representation:
- Volume particles for MPM physics simulation
- Surface particles for Gaussian Splatting rendering
"""

import open3d as o3d
import numpy as np
import torch
from typing import Tuple, Optional, NamedTuple
from pathlib import Path


class BasicPointCloud(NamedTuple):
    """Point cloud data structure compatible with Gaussian Splatting"""
    points: np.ndarray   # (N, 3) positions
    colors: np.ndarray   # (N, 3) RGB values [0, 1]
    normals: np.ndarray  # (N, 3) surface normals


class MeshToPointCloudConverter:
    """
    Convert 3D mesh to dual particle representation:
    - Volumetric particles for MPM simulation
    - Surface particles for Gaussian Splat rendering
    """

    def __init__(
        self,
        mesh_path: str,
        target_particle_count: int = 30000,
        surface_sample_ratio: float = 0.4,
        use_poisson: bool = False,
        poisson_depth: int = 8,
        normalize_to_unit_cube: bool = True
    ):
        """
        Args:
            mesh_path: Path to .obj or .ply mesh file
            target_particle_count: Total particles (surface + volume)
            surface_sample_ratio: Fraction of particles on surface (default 40%)
            use_poisson: Use Poisson disk sampling (experimental)
            poisson_depth: Octree depth for Poisson reconstruction
            normalize_to_unit_cube: Normalize mesh to [0, 1]³ space
        """
        self.mesh_path = Path(mesh_path)
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        self.target_count = target_particle_count
        self.surface_ratio = surface_sample_ratio
        self.use_poisson = use_poisson
        self.poisson_depth = poisson_depth
        self.normalize = normalize_to_unit_cube

        print(f"[MeshConverter] Initialized")
        print(f"  - Mesh: {self.mesh_path.name}")
        print(f"  - Target particles: {target_particle_count}")
        print(f"  - Surface ratio: {surface_sample_ratio:.1%}")

    def load_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Load mesh from file and optionally normalize to unit cube

        Returns:
            Normalized triangle mesh with vertex normals
        """
        print(f"[MeshConverter] Loading mesh from {self.mesh_path}")

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))

        if not mesh.has_vertices():
            raise ValueError(f"Mesh has no vertices: {self.mesh_path}")

        # Compute normals if missing
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        print(f"  - Original vertices: {len(vertices)}")
        print(f"  - Triangles: {len(mesh.triangles)}")

        if self.normalize:
            # Normalize to unit cube [0, 1]³
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            scale = (bbox_max - bbox_min).max()

            # Normalize and fit to 95% of unit cube (avoid boundaries)
            vertices = (vertices - center) / scale * 0.95 + 0.5
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            print(f"  - Normalized to [0, 1]³ (95% fill)")
            print(f"  - Original center: {center}")
            print(f"  - Original scale: {scale:.4f}")

        return mesh

    def sample_surface_particles(
        self,
        mesh: o3d.geometry.TriangleMesh,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample points uniformly on mesh surface

        Args:
            mesh: Input triangle mesh
            n_samples: Number of surface points to sample

        Returns:
            points: (n_samples, 3) surface positions
            colors: (n_samples, 3) RGB colors
            normals: (n_samples, 3) surface normals
        """
        print(f"[MeshConverter] Sampling {n_samples} surface particles...")

        # Uniform surface sampling
        pcd = mesh.sample_points_uniformly(number_of_points=n_samples)

        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        # Get colors from mesh or generate from position
        if mesh.has_vertex_colors():
            # Interpolate colors from mesh vertices
            colors = self._interpolate_vertex_colors(mesh, points)
        else:
            # Generate gray gradient (neutral for red crack visibility)
            # Use Y-coordinate (height) for gradient: dark gray (bottom) → light gray (top)
            gray_values = 0.3 + 0.4 * points[:, 1]  # Y: 0.3-0.7 (dark to light)
            colors = np.stack([gray_values, gray_values, gray_values], axis=1)
            colors = np.clip(colors, 0.0, 1.0)

        print(f"  - Sampled {len(points)} surface points")
        print(f"  - Position range: [{points.min():.3f}, {points.max():.3f}]")

        return points, colors, normals

    def sample_volumetric_particles(
        self,
        mesh: o3d.geometry.TriangleMesh,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points inside mesh volume using rejection sampling

        Args:
            mesh: Input triangle mesh
            n_samples: Number of volume points to sample

        Returns:
            points: (n_samples, 3) interior positions
            colors: (n_samples, 3) RGB colors (default gray)
        """
        print(f"[MeshConverter] Sampling {n_samples} volumetric particles...")

        if self.use_poisson:
            pcd = self._poisson_volume_sample(mesh, n_samples)
        else:
            pcd = self._rejection_sample_volume(mesh, n_samples)

        points = np.asarray(pcd.points)
        # Generate gray gradient (slightly darker than surface for interior)
        gray_values = 0.25 + 0.35 * points[:, 1]  # Y: 0.25-0.6 (darker gray)
        colors = np.stack([gray_values, gray_values, gray_values], axis=1)
        colors = np.clip(colors, 0.0, 1.0)

        print(f"  - Sampled {len(points)} volume points")

        return points, colors

    def _rejection_sample_volume(
        self,
        mesh: o3d.geometry.TriangleMesh,
        n_samples: int
    ) -> o3d.geometry.PointCloud:
        """
        Rejection sampling: generate random points, keep those inside mesh

        Uses raycasting to determine if points are inside mesh
        """
        # Create raycasting scene
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        points = []
        batch_size = n_samples * 3  # Oversample for efficiency
        attempts = 0
        max_attempts = 10

        while len(points) < n_samples and attempts < max_attempts:
            # Generate random candidates in [0, 1]³
            candidates = np.random.rand(batch_size, 3).astype(np.float32)

            # Check if inside mesh using raycasting
            query_points = o3d.core.Tensor(candidates, dtype=o3d.core.Dtype.Float32)
            occupancy = scene.compute_occupancy(query_points).numpy()

            # Keep interior points
            inside = candidates[occupancy > 0.5]
            points.extend(inside)

            attempts += 1
            if attempts > 1:
                print(f"    Attempt {attempts}: {len(points)}/{n_samples} points")

        # Trim to exact count
        points = np.array(points[:n_samples])

        if len(points) < n_samples:
            print(f"  Warning: Only sampled {len(points)}/{n_samples} volume points")
            print(f"    Mesh may not be watertight. Padding with surface-offset points.")

            # Fallback: sample remaining from surface with slight inward offset
            remaining = n_samples - len(points)
            surface_pcd = mesh.sample_points_uniformly(remaining)
            surface_pts = np.asarray(surface_pcd.points)
            normals = np.asarray(surface_pcd.normals)

            # Offset inward by small amount
            offset_pts = surface_pts - normals * 0.02  # 2% inward
            points = np.vstack([points, offset_pts])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _poisson_volume_sample(
        self,
        mesh: o3d.geometry.TriangleMesh,
        n_samples: int
    ) -> o3d.geometry.PointCloud:
        """
        Poisson reconstruction + voxel sampling (experimental)

        Creates watertight mesh then samples interior voxels
        """
        print(f"  - Using Poisson reconstruction (depth={self.poisson_depth})")

        # Get dense surface point cloud
        surface_pcd = mesh.sample_points_uniformly(n_samples * 3)

        # Poisson surface reconstruction
        mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            surface_pcd, depth=self.poisson_depth
        )

        # Remove low-density vertices (outside hull)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.1)
        vertices_to_remove = densities < density_threshold
        mesh_poisson.remove_vertices_by_mask(vertices_to_remove)

        # Voxelize and sample
        voxel_size = 1.0 / (2 ** self.poisson_depth * 2)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            mesh_poisson, voxel_size=voxel_size
        )

        # Extract voxel centers as volume particles
        voxels = voxel_grid.get_voxels()
        points = np.array([voxel_grid.get_voxel_center_coordinate(v.grid_index)
                          for v in voxels])

        # Subsample to target count
        if len(points) > n_samples:
            indices = np.random.choice(len(points), n_samples, replace=False)
            points = points[indices]
        elif len(points) < n_samples:
            print(f"  Warning: Poisson only generated {len(points)} points, padding...")
            # Pad with rejection sampling
            remaining_pcd = self._rejection_sample_volume(mesh, n_samples - len(points))
            remaining_points = np.asarray(remaining_pcd.points)
            points = np.vstack([points, remaining_points])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _interpolate_vertex_colors(
        self,
        mesh: o3d.geometry.TriangleMesh,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate vertex colors to sampled points (nearest neighbor)

        Args:
            mesh: Mesh with vertex colors
            points: Query points

        Returns:
            colors: (N, 3) interpolated RGB colors
        """
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors)

        # Build KDTree for nearest neighbor
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn.fit(vertices)

        distances, indices = nn.kneighbors(points)
        colors = vertex_colors[indices.flatten()]

        return colors

    def convert(self) -> Tuple[BasicPointCloud, BasicPointCloud, np.ndarray]:
        """
        Main conversion: mesh → (volume_pcd, surface_pcd, surface_mask)

        Returns:
            volume_pcd: All particles for MPM (N_total, 3)
            surface_pcd: Surface particles for Gaussian Splats (N_surf, 3)
            surface_mask: Boolean mask (N_total,) - True for surface particles
        """
        print(f"\n{'='*60}")
        print(f"Converting Mesh → Point Clouds")
        print(f"{'='*60}")

        # Load and normalize mesh
        mesh = self.load_mesh()

        # Calculate particle counts
        n_surface = int(self.target_count * self.surface_ratio)
        n_volume = self.target_count - n_surface

        print(f"\nParticle allocation:")
        print(f"  - Surface: {n_surface} ({self.surface_ratio:.1%})")
        print(f"  - Volume: {n_volume} ({1-self.surface_ratio:.1%})")
        print(f"  - Total: {self.target_count}")

        # Sample surface
        surf_pts, surf_colors, surf_normals = self.sample_surface_particles(
            mesh, n_surface
        )

        # Sample volume interior
        vol_pts, vol_colors = self.sample_volumetric_particles(mesh, n_volume)
        vol_normals = np.zeros_like(vol_pts)  # No normals for interior

        # Combine for MPM (all particles)
        all_points = np.vstack([surf_pts, vol_pts])
        all_colors = np.vstack([surf_colors, vol_colors])
        all_normals = np.vstack([surf_normals, vol_normals])

        volume_pcd = BasicPointCloud(
            points=all_points,
            colors=all_colors,
            normals=all_normals
        )

        # Surface only for Gaussian Splats
        surface_pcd = BasicPointCloud(
            points=surf_pts,
            colors=surf_colors,
            normals=surf_normals
        )

        # Surface mask: first n_surface particles are True
        surface_mask = np.zeros(self.target_count, dtype=bool)
        surface_mask[:n_surface] = True

        print(f"\nConversion complete!")
        print(f"  - Volume PCD: {all_points.shape[0]} particles")
        print(f"  - Surface PCD: {surf_pts.shape[0]} particles")
        print(f"  - Surface mask: {surface_mask.sum()} True values")
        print(f"{'='*60}\n")

        return volume_pcd, surface_pcd, surface_mask


def test_converter():
    """Simple test with sphere geometry"""
    print("Testing MeshToPointCloudConverter...")

    # Create a test sphere mesh
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
    mesh.compute_vertex_normals()

    # Save temporarily
    test_path = "test_sphere.ply"
    o3d.io.write_triangle_mesh(test_path, mesh)

    # Test converter
    converter = MeshToPointCloudConverter(
        mesh_path=test_path,
        target_particle_count=1000,
        surface_sample_ratio=0.4
    )

    volume_pcd, surface_pcd, mask = converter.convert()

    # Verify
    assert volume_pcd.points.shape[0] == 1000, "Incorrect total particle count"
    assert surface_pcd.points.shape[0] == 400, "Incorrect surface particle count"
    assert mask.sum() == 400, "Incorrect surface mask count"
    assert np.all(volume_pcd.points >= 0.0) and np.all(volume_pcd.points <= 1.0), \
        "Points outside [0, 1]³"

    print("✓ All tests passed!")

    # Clean up
    import os
    if os.path.exists(test_path):
        os.remove(test_path)


if __name__ == "__main__":
    test_converter()
