"""Gaussian crack visualization via damage-based color blending."""

import torch
from torch import Tensor

# SH coefficient C0 for 0th order
C0 = 0.28209479177387814


def _rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB [0,1] to 0th-order SH coefficient."""
    return (rgb - 0.5) / C0


class GaussianCrackVisualizer:
    """
    Visualizes cracks on Gaussian Splats by blending colors toward
    crack_color based on surface damage values.

    On first call with preserve_original=True, stores original SH features.
    Each subsequent call blends from originals:
        features_dc = original * (1-c) + red_sh * c
    """

    def __init__(
        self,
        crack_color=(0.8, 0.1, 0.1),
        opacity_mode="smooth",
        color_mode="overlay",
        scale_mode=None,
        damage_threshold=0.3,
        crack_brightness=5.0,
        device="cuda"
    ):
        self.crack_color = crack_color
        self.opacity_mode = opacity_mode
        self.color_mode = color_mode
        self.scale_mode = scale_mode
        self.damage_threshold = damage_threshold
        self.crack_brightness = crack_brightness
        self.device = device

        # Original features (stored on first update call)
        self._original_dc = None
        self._original_rest = None

        # Precompute red SH
        red_rgb = torch.tensor(
            [[crack_color[0], crack_color[1], crack_color[2]]],
            dtype=torch.float32
        )
        self._red_sh = _rgb_to_sh(red_rgb) * crack_brightness  # (1, 3)

        print(f"[GaussianCrackVisualizer] Initialized")
        print(f"  - Crack color: {crack_color}")
        print(f"  - Brightness: {crack_brightness}x")
        print(f"  - Device: {device}")

    def update_gaussians(
        self,
        gaussians,
        c_surface: Tensor,
        x_world: Tensor,
        preserve_original: bool = True
    ):
        """
        Update Gaussian properties based on surface damage.

        Blends Gaussian DC features toward bright red based on damage.
        Higher-order SH features are suppressed for damaged Gaussians.

        Args:
            gaussians: GaussianModel to update
            c_surface: Surface damage values (N_surf,) in [0, 1]
            x_world: Updated world positions (N_surf, 3)
            preserve_original: Store originals on first call
        """
        # Store originals on first call
        if preserve_original and self._original_dc is None:
            self._original_dc = gaussians._features_dc.data.clone()
            self._original_rest = gaussians._features_rest.data.clone()

        # Update positions
        gaussians._xyz.data = x_world

        # Blend colors based on damage
        c = c_surface.clamp(0.0, 1.0)  # (N_surf,)

        if c.max().item() > 0.01:
            # Blend factor per Gaussian: (N, 1, 1) for broadcasting
            blend = c.unsqueeze(1).unsqueeze(1)

            # Red SH on correct device: (1, 1, 3)
            red_sh = self._red_sh.to(device=c.device).unsqueeze(0)

            # Blend DC features: original * (1-c) + red * c
            original_dc = self._original_dc
            gaussians._features_dc.data = original_dc * (1.0 - blend) + red_sh * blend

            # Suppress higher-order SH for damaged regions
            original_rest = self._original_rest
            gaussians._features_rest.data = original_rest * (1.0 - blend)
