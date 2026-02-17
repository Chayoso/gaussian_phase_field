"""Gaussian crack visualization: physical displacement + opacity gap + edge darkening."""

import torch
from torch import Tensor

C0 = 0.28209479177387814  # SH 0th-order coefficient


def _rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB [0,1] to 0th-order SH coefficient."""
    return (rgb - 0.5) / C0


class GaussianCrackVisualizer:
    """
    Crack opening visualization on Gaussian Splats.

    Applies physical displacement of Gaussians near crack paths to create
    visible gaps, plus opacity/color effects for crack interior rendering.
    """

    def __init__(
        self,
        crack_color=(0.8, 0.1, 0.1),
        opacity_mode="smooth",
        color_mode="overlay",
        scale_mode=None,
        damage_threshold=0.3,
        crack_brightness=5.0,
        sharp_k=30.0,
        edge_width=0.15,
        crack_opacity_reduction=0.85,
        max_opening=0.015,
        gap_fraction=0.3,
        edge_darken=0.3,
        red_accent=0.15,
        device="cuda"
    ):
        self.crack_color = crack_color
        self.damage_threshold = damage_threshold
        self.crack_brightness = crack_brightness
        self.crack_opacity_reduction = crack_opacity_reduction
        self.device = device

        self.max_opening = max_opening
        self.gap_fraction = gap_fraction
        self.edge_darken = edge_darken
        self.red_accent = red_accent

        self._original_dc = None
        self._original_rest = None
        self._original_opacity = None
        self._original_scaling = None

        # Precompute SH coefficients for crack colors
        red_rgb = torch.tensor(
            [[crack_color[0], crack_color[1], crack_color[2]]],
            dtype=torch.float32
        )
        self._red_sh = _rgb_to_sh(red_rgb) * crack_brightness
        self._dark_sh = _rgb_to_sh(
            torch.tensor([[0.05, 0.02, 0.02]], dtype=torch.float32)
        ) * crack_brightness

        print(f"[GaussianCrackVisualizer] Initialized (crack opening mode)")
        print(f"  - Max opening: {max_opening} world units")
        print(f"  - Gap fraction: {gap_fraction}")
        print(f"  - Edge darken: {edge_darken}")
        print(f"  - Red accent: {red_accent}")
        print(f"  - Device: {device}")

    @torch.no_grad()
    def _crack_geometry(self, positions: Tensor, crack_paths: list, crack_width: float):
        """
        Compute distance and perpendicular direction from each Gaussian
        to the nearest crack polyline segment.

        Returns:
            min_dist: (N,) distance to nearest crack segment
            v_perp_hat: (N, 3) unit perpendicular direction (away from crack)
        """
        N = positions.shape[0]
        device = positions.device

        best_dist = torch.full((N,), float('inf'), device=device)
        best_closest = torch.zeros(N, 3, device=device)
        best_tangent = torch.zeros(N, 3, device=device)

        for path in crack_paths:
            if path.shape[0] < 2:
                dist = (positions - path[0]).norm(dim=1)
                closer = dist < best_dist
                if closer.any():
                    best_dist[closer] = dist[closer]
                    best_closest[closer] = path[0]
                continue

            a = path[:-1]
            b = path[1:]
            ab = b - a
            ab_len = ab.norm(dim=1)
            valid = ab_len > 1e-10
            if not valid.any():
                continue

            a_v = a[valid]
            ab_v = ab[valid]
            ab_len_v = ab_len[valid]
            ab_hat_v = ab_v / ab_len_v.unsqueeze(1)

            S = a_v.shape[0]
            MAX_BATCH = 2_000_000
            chunk_size = max(1, MAX_BATCH // max(S, 1))

            for ci in range(0, N, chunk_size):
                cj = min(ci + chunk_size, N)
                pos_chunk = positions[ci:cj]
                C = cj - ci

                ap = pos_chunk.unsqueeze(1) - a_v.unsqueeze(0)
                t = (ap * ab_hat_v.unsqueeze(0)).sum(dim=2)
                t = t.clamp(min=0.0)
                t = torch.minimum(t, ab_len_v.unsqueeze(0))

                closest = a_v.unsqueeze(0) + t.unsqueeze(2) * ab_hat_v.unsqueeze(0)
                diff = pos_chunk.unsqueeze(1) - closest
                dist = diff.norm(dim=2)
                min_dist_chunk, min_idx = dist.min(dim=1)

                closer = min_dist_chunk < best_dist[ci:cj]
                if closer.any():
                    c_idx = torch.arange(C, device=device)
                    win_idx = min_idx[closer]
                    best_dist[ci:cj][closer] = min_dist_chunk[closer]
                    best_closest[ci:cj][closer] = closest[c_idx[closer], win_idx]
                    best_tangent[ci:cj][closer] = ab_hat_v[win_idx]

        v = positions - best_closest
        v_dot_t = (v * best_tangent).sum(dim=1, keepdim=True)
        v_perp = v - v_dot_t * best_tangent
        v_perp_len = v_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
        v_perp_hat = v_perp / v_perp_len

        return best_dist, v_perp_hat

    @torch.no_grad()
    def _apply_crack_opening(self, gaussians, crack_paths, crack_width):
        """Apply physical crack opening: displacement + opacity gap + edge effects."""
        positions = gaussians._xyz.data
        N = positions.shape[0]
        device = positions.device

        min_dist, v_perp_hat = self._crack_geometry(positions, crack_paths, crack_width)
        norm_dist = (min_dist / crack_width).clamp(0.0)

        affected = norm_dist < 1.2
        if not affected.any():
            return

        # 1. Edge displacement
        edge_zone = (norm_dist >= 0.4) & (norm_dist < 1.0)
        opening_mag = torch.zeros(N, device=device)
        if edge_zone.any():
            t_e = ((norm_dist[edge_zone] - 0.4) / 0.6).clamp(0.0, 1.0)
            opening_mag[edge_zone] = self.max_opening * t_e * (1.0 - t_e) * 4.0
        gaussians._xyz.data = positions + v_perp_hat * opening_mag.unsqueeze(1)

        # 2. Dark interior line (cubic falloff)
        dark_sh = self._dark_sh.to(device).unsqueeze(0)
        red_sh = self._red_sh.to(device).unsqueeze(0)

        darken_amount = torch.zeros(N, device=device)
        in_crack = norm_dist < 1.0
        if in_crack.any():
            t = norm_dist[in_crack].clamp(0.0, 1.0)
            darken_amount[in_crack] = (1.0 - t) ** 3

        d = darken_amount.unsqueeze(1).unsqueeze(1)
        current_dc = gaussians._features_dc.data
        gaussians._features_dc.data = (1.0 - d) * current_dc + d * dark_sh
        current_rest = gaussians._features_rest.data
        gaussians._features_rest.data = current_rest * (1.0 - d)

        # 3. Scale shrinkage at crack center
        scale_factor = torch.ones(N, device=device)
        narrow_zone = norm_dist < 0.6
        if narrow_zone.any():
            t_s = (norm_dist[narrow_zone] / 0.6).clamp(0.0, 1.0)
            scale_factor[narrow_zone] = 0.4 + 0.6 * t_s
        gaussians._scaling.data = gaussians._scaling.data + torch.log(
            scale_factor.unsqueeze(1).clamp(min=0.1)
        )

        # 4. Center opacity dip
        opacity_factor = torch.ones(N, device=device)
        center = norm_dist < 0.1
        if center.any():
            t_c = (norm_dist[center] / 0.1).clamp(0.0, 1.0)
            opacity_factor[center] = 0.5 + 0.5 * t_c
        current_prob = torch.sigmoid(gaussians._opacity.data)
        new_prob = (current_prob * opacity_factor.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
        gaussians._opacity.data = torch.log(new_prob / (1.0 - new_prob))

        # 5. Red accent at boundary
        if self.red_accent > 0:
            red_zone = (norm_dist >= 0.75) & (norm_dist < 1.05)
            red_amount = torch.zeros(N, device=device)
            if red_zone.any():
                t_r = ((norm_dist[red_zone] - 0.9) / 0.15).clamp(-1.0, 1.0)
                red_amount[red_zone] = self.red_accent * (1.0 - t_r * t_r)
            r = red_amount.unsqueeze(1).unsqueeze(1)
            dc = gaussians._features_dc.data
            gaussians._features_dc.data = (1.0 - r) * dc + r * red_sh

    def update_gaussians(
        self,
        gaussians,
        c_surface: Tensor,
        x_world: Tensor,
        preserve_original: bool = True,
        crack_paths: list = None,
        crack_width: float = 0.03
    ):
        """Update Gaussian properties for crack visualization."""
        if preserve_original and self._original_dc is None:
            self._original_dc = gaussians._features_dc.data.clone()
            self._original_rest = gaussians._features_rest.data.clone()
            self._original_opacity = gaussians._opacity.data.clone()
            self._original_scaling = gaussians._scaling.data.clone()

        gaussians._xyz.data = x_world

        gaussians._features_dc.data.copy_(self._original_dc)
        gaussians._features_rest.data.copy_(self._original_rest)
        gaussians._opacity.data.copy_(self._original_opacity)
        gaussians._scaling.data.copy_(self._original_scaling)

        has_polylines = (crack_paths is not None
                         and len(crack_paths) > 0
                         and any(p.shape[0] >= 2 for p in crack_paths))

        if has_polylines:
            self._apply_crack_opening(gaussians, crack_paths, crack_width)
