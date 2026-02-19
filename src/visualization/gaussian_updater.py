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

        # 2. Dark interior line (DISABLED)
        # dark_sh = self._dark_sh.to(device).unsqueeze(0)
        # red_sh = self._red_sh.to(device).unsqueeze(0)
        # darken_amount = torch.zeros(N, device=device)
        # in_crack = norm_dist < 1.0
        # if in_crack.any():
        #     t = norm_dist[in_crack].clamp(0.0, 1.0)
        #     darken_amount[in_crack] = (1.0 - t) ** 3
        # d = darken_amount.unsqueeze(1).unsqueeze(1)
        # current_dc = gaussians._features_dc.data
        # gaussians._features_dc.data = (1.0 - d) * current_dc + d * dark_sh
        # current_rest = gaussians._features_rest.data
        # gaussians._features_rest.data = current_rest * (1.0 - d)

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

        # 5. Red accent at boundary (DISABLED)
        # if self.red_accent > 0:
        #     red_zone = (norm_dist >= 0.75) & (norm_dist < 1.05)
        #     red_amount = torch.zeros(N, device=device)
        #     if red_zone.any():
        #         t_r = ((norm_dist[red_zone] - 0.9) / 0.15).clamp(-1.0, 1.0)
        #         red_amount[red_zone] = self.red_accent * (1.0 - t_r * t_r)
        #     r = red_amount.unsqueeze(1).unsqueeze(1)
        #     dc = gaussians._features_dc.data
        #     gaussians._features_dc.data = (1.0 - r) * dc + r * red_sh

    @torch.no_grad()
    def _apply_hybrid_crack(self, gaussians, c_surface: Tensor,
                            crack_paths: list, crack_width: float):
        """Hybrid crack visualization: polyline shape × c_surface intensity.

        - Polyline proximity gives sharp, branching crack geometry
        - c_surface gates the effect so only physically damaged regions are affected
        - Eliminates artifacts from stale paths (low c_surface → no effect)

        Effect strength = polyline_proximity × c_surface_damage
        """
        positions = gaussians._xyz.data
        N = positions.shape[0]
        device = positions.device

        # 1. Compute polyline proximity: sharp crack shape
        min_dist, v_perp_hat = self._crack_geometry(positions, crack_paths, crack_width)
        norm_dist = (min_dist / crack_width).clamp(0.0)

        # Proximity factor: 1.0 at crack center → 0.0 beyond crack_width
        prox = (1.0 - norm_dist).clamp(0.0, 1.0)

        # 2. Combined strength: polyline shape × phase field damage
        #    Close to polyline guarantees a minimum effect (crack tip visibility)
        #    c_surface amplifies effect where damage is confirmed
        min_prox_effect = 0.4 * prox  # polyline alone gives 40% effect
        damage_boost = 0.6 * prox * c_surface  # damage adds remaining 60%
        strength = min_prox_effect + damage_boost  # (N,)

        affected = strength > 0.05
        if not affected.any():
            return

        # 3. Edge displacement — push Gaussians away from crack center
        #    Only where both proximity and damage are significant
        disp_mask = (norm_dist >= 0.3) & (norm_dist < 1.0) & (c_surface > self.damage_threshold)
        if disp_mask.any():
            t_e = ((norm_dist[disp_mask] - 0.3) / 0.7).clamp(0.0, 1.0)
            # Bell curve displacement, gated by damage
            mag = self.max_opening * t_e * (1.0 - t_e) * 4.0 * c_surface[disp_mask]
            disp = v_perp_hat[disp_mask] * mag.unsqueeze(1)
            gaussians._xyz.data[disp_mask] = positions[disp_mask] + disp

        # 4. Scale shrinkage — create visible gap along crack line
        scale_mask = strength > 0.1
        if scale_mask.any():
            s = strength[scale_mask].clamp(0.0, 1.0)
            # scale: 1.0 → 0.2 as strength increases
            scale_mult = 1.0 - 0.8 * (s ** 1.5)
            gaussians._scaling.data[scale_mask] += torch.log(
                scale_mult.unsqueeze(1).clamp(min=0.01))

        # 5. Opacity reduction — fade crack interior
        opa_mask = strength > 0.3
        if opa_mask.any():
            t_o = ((strength[opa_mask] - 0.3) / 0.7).clamp(0.0, 1.0)
            opacity_mult = 1.0 - 0.85 * (t_o ** 2)
            cur_prob = torch.sigmoid(gaussians._opacity.data[opa_mask])
            new_prob = (cur_prob * opacity_mult.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
            gaussians._opacity.data[opa_mask] = torch.log(new_prob / (1.0 - new_prob))

    @torch.no_grad()
    def _apply_damage_visualization(self, gaussians, c_surface: Tensor):
        """Phase-field damage visualization (post-fragmentation).

        After fragments physically separate, the gap IS the crack.
        We only need to show material degradation on each fragment's surface:
        - Scale shrinkage where damage is significant → thinner splats at cracks
        - Opacity reduction at high damage → transparency at fully broken areas

        Uses smooth cubic falloff to avoid hard boundaries.
        """
        N = c_surface.shape[0]
        device = c_surface.device
        thresh = self.damage_threshold

        # 1. Scale shrinkage: starts at lower threshold for gradual onset
        low_thresh = thresh * 0.5  # Start showing effect earlier
        damaged = c_surface > low_thresh
        if damaged.any():
            t = ((c_surface[damaged] - low_thresh)
                 / (1.0 - low_thresh)).clamp(0.0, 1.0)
            # Smooth cubic: gradual onset, strong at high damage
            scale_mult = 1.0 - 0.9 * (t ** 2) * (3.0 - 2.0 * t)
            gaussians._scaling.data[damaged] += torch.log(
                scale_mult.unsqueeze(1).clamp(min=0.01))

        # 2. Opacity reduction: stronger at high damage
        opa_thresh = thresh
        high_damage = c_surface > opa_thresh
        if high_damage.any():
            t_o = ((c_surface[high_damage] - opa_thresh)
                   / (1.0 - opa_thresh)).clamp(0.0, 1.0)
            # Aggressive fade: fully broken material should be transparent
            opacity_mult = 1.0 - 0.95 * (t_o ** 2)
            cur_prob = torch.sigmoid(gaussians._opacity.data[high_damage])
            new_prob = (cur_prob * opacity_mult.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
            gaussians._opacity.data[high_damage] = torch.log(new_prob / (1.0 - new_prob))

    def update_gaussians(
        self,
        gaussians,
        c_surface: Tensor,
        x_world: Tensor,
        preserve_original: bool = True,
        crack_paths: list = None,
        crack_width: float = 0.03,
        debris_mask: Tensor = None,
    ):
        """Update Gaussian properties for crack visualization.

        Args:
            debris_mask: (N_surf,) bool — Gaussians belonging to small
                fragments.  These are hidden (opacity → 0) before any
                crack effects to avoid edge artifacts.
        """
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

        # Hide debris BEFORE crack effects to prevent edge artifacts
        if debris_mask is not None and debris_mask.any():
            gaussians._opacity.data[debris_mask] = -20.0
            gaussians._scaling.data[debris_mask] -= 5.0  # Shrink to ~0

        has_polylines = (crack_paths is not None
                         and len(crack_paths) > 0
                         and any(p.shape[0] >= 2 for p in crack_paths))
        has_damage = (c_surface is not None
                      and c_surface.max() > self.damage_threshold)

        if has_polylines and has_damage:
            # Pre-fragmentation: polyline shape × phase field intensity
            self._apply_hybrid_crack(gaussians, c_surface, crack_paths, crack_width)
        elif has_damage:
            # Post-fragmentation: c_surface only (physical gaps are the cracks)
            self._apply_damage_visualization(gaussians, c_surface)
