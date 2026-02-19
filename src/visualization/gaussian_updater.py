"""Gaussian crack visualization via AT2 phase-field damage."""

import torch
from torch import Tensor


class GaussianCrackVisualizer:
    """
    Crack visualization on Gaussian Splats using AT2 phase-field damage.

    Maps c_surface ∈ [0,1] (from AT2 PDE) to Gaussian scale and opacity.
    Polylines drive crack propagation physics; c_surface drives rendering.
    """

    def __init__(
        self,
        damage_threshold: float = 0.3,
        device: str = "cuda",
    ):
        self.damage_threshold = damage_threshold
        self.device = device

        self._original_dc = None
        self._original_rest = None
        self._original_opacity = None
        self._original_scaling = None

        print(f"[GaussianCrackVisualizer] Initialized (AT2 damage mode)")
        print(f"  - Damage threshold: {damage_threshold}")
        print(f"  - Device: {device}")

    @torch.no_grad()
    def _apply_damage_visualization(self, gaussians, c_surface: Tensor):
        """Visualize AT2 damage field on Gaussian Splats.

        Maps c_surface → scale shrinkage + opacity reduction.
        Smooth cubic falloff avoids hard seams at damage boundaries.
        """
        thresh = self.damage_threshold

        # Scale shrinkage: gradual onset from thresh*0.5
        low_thresh = thresh * 0.5
        damaged = c_surface > low_thresh
        if damaged.any():
            t = ((c_surface[damaged] - low_thresh)
                 / (1.0 - low_thresh)).clamp(0.0, 1.0)
            scale_mult = 1.0 - 0.9 * (t ** 2) * (3.0 - 2.0 * t)  # cubic
            gaussians._scaling.data[damaged] += torch.log(
                scale_mult.unsqueeze(1).clamp(min=0.01))

        # Opacity reduction: aggressive fade at high damage
        high_damage = c_surface > thresh
        if high_damage.any():
            t_o = ((c_surface[high_damage] - thresh)
                   / (1.0 - thresh)).clamp(0.0, 1.0)
            opacity_mult = 1.0 - 0.95 * (t_o ** 2)
            cur_prob = torch.sigmoid(gaussians._opacity.data[high_damage])
            new_prob = (cur_prob * opacity_mult.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
            gaussians._opacity.data[high_damage] = torch.log(
                new_prob / (1.0 - new_prob))

    def update_gaussians(
        self,
        gaussians,
        c_surface: Tensor,
        x_world: Tensor,
        preserve_original: bool = True,
        debris_mask: Tensor = None,
    ):
        """Update Gaussian properties each frame.

        Args:
            gaussians:        3DGS Gaussian model
            c_surface:        (N_surf,) AT2 damage values ∈ [0,1]
            x_world:          (N_surf, 3) current surface positions in world space
            preserve_original: cache original Gaussian properties on first call
            debris_mask:      (N_surf,) bool — small fragment Gaussians to hide
        """
        # Cache original properties once
        if preserve_original and self._original_dc is None:
            self._original_dc = gaussians._features_dc.data.clone()
            self._original_rest = gaussians._features_rest.data.clone()
            self._original_opacity = gaussians._opacity.data.clone()
            self._original_scaling = gaussians._scaling.data.clone()

        # Update positions from MPM physics
        gaussians._xyz.data = x_world

        # Restore originals before applying effects
        gaussians._features_dc.data.copy_(self._original_dc)
        gaussians._features_rest.data.copy_(self._original_rest)
        gaussians._opacity.data.copy_(self._original_opacity)
        gaussians._scaling.data.copy_(self._original_scaling)

        # Hide debris before crack effects (prevents edge artifacts)
        if debris_mask is not None and debris_mask.any():
            gaussians._opacity.data[debris_mask] = -20.0
            gaussians._scaling.data[debris_mask] -= 5.0

        # Apply damage visualization
        has_damage = (c_surface is not None
                      and c_surface.max() > self.damage_threshold)
        if has_damage:
            self._apply_damage_visualization(gaussians, c_surface)
