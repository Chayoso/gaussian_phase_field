"""
Coordinate Mapper

Bidirectional mapping between MPM grid space [0,1]³ and world space.
MPM simulation operates in normalized [0,1]³ space, while Gaussian Splats
use arbitrary world space coordinates.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Union


class CoordinateMapper:
    """
    Bidirectional coordinate transformation:
    - MPM space: [0, 1]³ normalized cube
    - World space: Arbitrary scale and center

    Transform:
        x_world = (x_mpm - 0.5) * world_scale + world_center
        x_mpm = (x_world - world_center) / world_scale + 0.5
    """

    def __init__(
        self,
        mpm_bounds: Tuple[float, float] = (0.0, 1.0),
        world_center: Union[np.ndarray, list] = None,
        world_scale: float = 2.0,
        device: torch.device = None
    ):
        """
        Args:
            mpm_bounds: (min, max) for MPM simulation domain (default [0, 1])
            world_center: Center of world space bounding box (default [0, 0, 0])
            world_scale: Size of world space bounding box (default 2.0)
            device: PyTorch device for tensor operations
        """
        self.mpm_min, self.mpm_max = mpm_bounds
        self.mpm_range = self.mpm_max - self.mpm_min

        if world_center is None:
            world_center = np.array([0.0, 0.0, 0.0])
        else:
            world_center = np.array(world_center)

        self.world_center = torch.tensor(world_center, dtype=torch.float32)
        self.world_scale = world_scale
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move center to device
        self.world_center = self.world_center.to(self.device)

        print(f"[CoordinateMapper] Initialized")
        print(f"  - MPM bounds: [{self.mpm_min}, {self.mpm_max}]")
        print(f"  - World center: {world_center}")
        print(f"  - World scale: {world_scale}")
        print(f"  - Device: {self.device}")

    def mpm_to_world(self, x_mpm: Tensor) -> Tensor:
        """
        Convert MPM normalized coordinates → world space

        Args:
            x_mpm: (N, 3) or (..., 3) positions in [0, 1]³

        Returns:
            x_world: Positions in world space
        """
        device = x_mpm.device
        center = self.world_center.to(device)

        # Center to [-0.5, 0.5], scale, translate
        x_centered = x_mpm - 0.5
        x_world = x_centered * self.world_scale + center

        return x_world

    def world_to_mpm(self, x_world: Tensor) -> Tensor:
        """
        Convert world space → MPM normalized coordinates [0, 1]³

        Args:
            x_world: (N, 3) or (..., 3) positions in world space

        Returns:
            x_mpm: Positions in MPM space [0, 1]³
        """
        device = x_world.device
        center = self.world_center.to(device)

        # Inverse transformation
        x_centered = (x_world - center) / self.world_scale
        x_mpm = x_centered + 0.5

        # Clamp to MPM bounds (safety)
        x_mpm = x_mpm.clamp(self.mpm_min, self.mpm_max)

        return x_mpm

    def velocity_mpm_to_world(self, v_mpm: Tensor) -> Tensor:
        """
        Convert MPM velocity → world space velocity

        Note: Only scaling applies (no translation for velocities)

        Args:
            v_mpm: (N, 3) velocities in MPM space

        Returns:
            v_world: Velocities in world space
        """
        return v_mpm * self.world_scale

    def velocity_world_to_mpm(self, v_world: Tensor) -> Tensor:
        """
        Convert world space velocity → MPM velocity

        Args:
            v_world: (N, 3) velocities in world space

        Returns:
            v_mpm: Velocities in MPM space
        """
        return v_world / self.world_scale

    def scale_mpm_to_world(self, s_mpm: float) -> float:
        """Convert scalar distance from MPM → world space"""
        return s_mpm * self.world_scale

    def scale_world_to_mpm(self, s_world: float) -> float:
        """Convert scalar distance from world → MPM space"""
        return s_world / self.world_scale

    def get_world_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get world space bounding box

        Returns:
            bbox_min: (3,) minimum corner
            bbox_max: (3,) maximum corner
        """
        center = self.world_center.cpu().numpy()
        half_scale = self.world_scale / 2

        bbox_min = center - half_scale
        bbox_max = center + half_scale

        return bbox_min, bbox_max

    def to(self, device: torch.device):
        """Move mapper to specified device"""
        self.device = device
        self.world_center = self.world_center.to(device)
        return self

    def __repr__(self) -> str:
        return (f"CoordinateMapper(mpm=[{self.mpm_min}, {self.mpm_max}], "
                f"world_center={self.world_center.cpu().numpy()}, "
                f"world_scale={self.world_scale})")


def test_coordinate_mapper():
    """Test bidirectional coordinate mapping"""
    print("Testing CoordinateMapper...")

    mapper = CoordinateMapper(
        world_center=[0, 0, 0],
        world_scale=2.0
    )

    # Test 1: Bidirectional consistency
    x_mpm = torch.rand(100, 3)  # Random points in [0, 1]³
    x_world = mapper.mpm_to_world(x_mpm)
    x_recovered = mapper.world_to_mpm(x_world)

    error = (x_mpm - x_recovered).abs().max().item()
    assert error < 1e-6, f"Bidirectional error too large: {error}"
    print(f"✓ Bidirectional consistency: max error = {error:.2e}")

    # Test 2: Known transformations
    # Center of MPM space (0.5, 0.5, 0.5) → world origin
    center_mpm = torch.tensor([[0.5, 0.5, 0.5]])
    center_world = mapper.mpm_to_world(center_mpm)
    assert torch.allclose(center_world, torch.zeros(1, 3), atol=1e-6), \
        "Center mapping incorrect"
    print(f"✓ Center mapping: MPM [0.5, 0.5, 0.5] → World {center_world.squeeze().tolist()}")

    # Test 3: Corner transformations
    corners_mpm = torch.tensor([[0, 0, 0], [1, 1, 1]])
    corners_world = mapper.mpm_to_world(corners_mpm)
    expected = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
    assert torch.allclose(corners_world, expected, atol=1e-6), \
        "Corner mapping incorrect"
    print(f"✓ Corner mapping: MPM [0,0,0] → World {corners_world[0].tolist()}")
    print(f"                   MPM [1,1,1] → World {corners_world[1].tolist()}")

    # Test 4: Velocity scaling
    v_mpm = torch.tensor([[1.0, 0.0, 0.0]])  # 1 MPM unit/s
    v_world = mapper.velocity_mpm_to_world(v_mpm)
    assert torch.allclose(v_world, torch.tensor([[2.0, 0.0, 0.0]])), \
        "Velocity scaling incorrect"
    print(f"✓ Velocity scaling: {v_mpm.squeeze().tolist()} → {v_world.squeeze().tolist()}")

    # Test 5: GPU compatibility
    if torch.cuda.is_available():
        mapper_gpu = mapper.to(torch.device('cuda'))
        x_mpm_gpu = x_mpm.cuda()
        x_world_gpu = mapper_gpu.mpm_to_world(x_mpm_gpu)
        assert x_world_gpu.is_cuda, "GPU mapping failed"
        print(f"✓ GPU compatibility: tensors on {x_world_gpu.device}")

    print("✓ All tests passed!\n")


if __name__ == "__main__":
    test_coordinate_mapper()
