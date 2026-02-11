"""Gaussian crack visualization: physical displacement + opacity gap + edge darkening."""

import torch
from torch import Tensor

# SH coefficient C0 for 0th order
C0 = 0.28209479177387814


def _rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB [0,1] to 0th-order SH coefficient."""
    return (rgb - 0.5) / C0


class GaussianCrackVisualizer:
    """
    Crack opening visualization on Gaussian Splats.

    Primary effect: physical displacement of Gaussians to create visible gaps.
    Gaussians on opposite sides of a crack path are pushed apart, creating
    a real gap where the background shows through.

    Secondary effects:
    - Opacity reduction at crack center (gap)
    - Edge darkening (simulates shadow/depth inside crack)
    - Subtle red accent at crack boundary
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

        # Crack opening parameters
        self.max_opening = max_opening
        self.gap_fraction = gap_fraction
        self.edge_darken = edge_darken
        self.red_accent = red_accent

        # Original features (stored on first update call)
        self._original_dc = None
        self._original_rest = None
        self._original_opacity = None
        self._original_scaling = None

        # KNN graph for fallback boundary visualization
        self._knn_idx = None
        self._knn_k = 10

        # Precompute SH coefficients
        red_rgb = torch.tensor(
            [[crack_color[0], crack_color[1], crack_color[2]]],
            dtype=torch.float32
        )
        self._red_sh = _rgb_to_sh(red_rgb) * crack_brightness  # (1, 3)

        dark_rgb = torch.tensor([[0.05, 0.02, 0.02]], dtype=torch.float32)
        self._dark_sh = _rgb_to_sh(dark_rgb) * crack_brightness  # (1, 3)

        print(f"[GaussianCrackVisualizer] Initialized (crack opening mode)")
        print(f"  - Max opening: {max_opening} world units")
        print(f"  - Gap fraction: {gap_fraction}")
        print(f"  - Edge darken: {edge_darken}")
        print(f"  - Red accent: {red_accent}")
        print(f"  - Device: {device}")

    @torch.no_grad()
    def _crack_geometry(
        self,
        positions: Tensor,
        crack_paths: list,
        crack_width: float
    ):
        """
        Compute crack geometry for all Gaussians.

        For each Gaussian, find the closest point on any crack polyline segment,
        then compute the perpendicular displacement direction.

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
                # Single point: Euclidean distance, no tangent
                dist = (positions - path[0]).norm(dim=1)
                closer = dist < best_dist
                if closer.any():
                    best_dist[closer] = dist[closer]
                    best_closest[closer] = path[0]
                continue

            a = path[:-1]    # (S, 3) segment starts
            b = path[1:]     # (S, 3) segment ends
            ab = b - a       # (S, 3)
            ab_len = ab.norm(dim=1)  # (S,)
            valid = ab_len > 1e-10

            if not valid.any():
                continue

            a_v = a[valid]           # (S', 3)
            ab_v = ab[valid]         # (S', 3)
            ab_len_v = ab_len[valid] # (S',)
            ab_hat_v = ab_v / ab_len_v.unsqueeze(1)  # (S', 3) unit tangents

            S = a_v.shape[0]
            MAX_BATCH = 2_000_000
            chunk_size = max(1, MAX_BATCH // max(S, 1))

            for ci in range(0, N, chunk_size):
                cj = min(ci + chunk_size, N)
                pos_chunk = positions[ci:cj]  # (C, 3)
                C = cj - ci

                # (C, S', 3): vector from each segment start to each Gaussian
                ap = pos_chunk.unsqueeze(1) - a_v.unsqueeze(0)

                # Project onto segment: t = dot(ap, ab_hat) clamped to [0, len]
                t = (ap * ab_hat_v.unsqueeze(0)).sum(dim=2)  # (C, S')
                t = t.clamp(min=0.0)
                t = torch.minimum(t, ab_len_v.unsqueeze(0))

                # Closest point on segment
                closest = a_v.unsqueeze(0) + t.unsqueeze(2) * ab_hat_v.unsqueeze(0)  # (C, S', 3)

                # Distance
                diff = pos_chunk.unsqueeze(1) - closest  # (C, S', 3)
                dist = diff.norm(dim=2)  # (C, S')

                # Find minimum segment per Gaussian
                min_dist_chunk, min_idx = dist.min(dim=1)  # (C,), (C,)

                # Update best where this path is closer
                closer = min_dist_chunk < best_dist[ci:cj]
                if closer.any():
                    c_idx = torch.arange(C, device=device)
                    win_idx = min_idx[closer]
                    best_dist[ci:cj][closer] = min_dist_chunk[closer]
                    best_closest[ci:cj][closer] = closest[c_idx[closer], win_idx]
                    best_tangent[ci:cj][closer] = ab_hat_v[win_idx]

        # Compute perpendicular direction: v - (v.t)*t
        v = positions - best_closest  # (N, 3)
        v_dot_t = (v * best_tangent).sum(dim=1, keepdim=True)  # (N, 1)
        v_perp = v - v_dot_t * best_tangent  # (N, 3)

        v_perp_len = v_perp.norm(dim=1, keepdim=True).clamp(min=1e-8)
        v_perp_hat = v_perp / v_perp_len

        return best_dist, v_perp_hat

    @torch.no_grad()
    def _apply_crack_opening(self, gaussians, crack_paths, crack_width):
        """
        Apply physical crack opening: displacement + opacity gap + edge effects.
        """
        positions = gaussians._xyz.data  # (N, 3)
        N = positions.shape[0]
        device = positions.device

        # Compute crack geometry
        min_dist, v_perp_hat = self._crack_geometry(positions, crack_paths, crack_width)

        # Normalized distance: 0 at crack center, 1 at boundary
        norm_dist = (min_dist / crack_width).clamp(0.0)  # (N,)

        # Only process Gaussians within influence zone
        affected = norm_dist < 1.2
        if not affected.any():
            return

        # --- 1. Subtle edge displacement (very conservative) ---
        edge_zone = (norm_dist >= 0.4) & (norm_dist < 1.0)
        opening_mag = torch.zeros(N, device=device)
        if edge_zone.any():
            t_e = ((norm_dist[edge_zone] - 0.4) / 0.6).clamp(0.0, 1.0)
            opening_mag[edge_zone] = self.max_opening * t_e * (1.0 - t_e) * 4.0
        gaussians._xyz.data = positions + v_perp_hat * opening_mag.unsqueeze(1)

        # --- 2. Dark interior line (sharp profile) ---
        dark_sh = self._dark_sh.to(device).unsqueeze(0)  # (1, 1, 3)
        red_sh = self._red_sh.to(device).unsqueeze(0)     # (1, 1, 3)

        darken_amount = torch.zeros(N, device=device)
        in_crack = norm_dist < 1.0
        if in_crack.any():
            t = norm_dist[in_crack].clamp(0.0, 1.0)
            # Sharp cubic falloff: fully dark at center, sharp transition at edge
            darken_amount[in_crack] = (1.0 - t) * (1.0 - t) * (1.0 - t)

        d = darken_amount.unsqueeze(1).unsqueeze(1)
        gaussians._features_dc.data = (1.0 - d) * self._original_dc + d * dark_sh

        # Suppress SH inside crack (flat dark, no specular)
        gaussians._features_rest.data = self._original_rest * (1.0 - d)

        # --- 3. Scale shrinkage at crack center (thinner Gaussians → sharper line) ---
        if self._original_scaling is not None:
            scale_factor = torch.ones(N, device=device)
            narrow_zone = norm_dist < 0.6
            if narrow_zone.any():
                t_s = (norm_dist[narrow_zone] / 0.6).clamp(0.0, 1.0)
                # Shrink to 40% at center, full size at 0.6
                scale_factor[narrow_zone] = 0.4 + 0.6 * t_s
            # Apply in log-space (scaling is stored as log)
            gaussians._scaling.data = self._original_scaling + torch.log(
                scale_factor.unsqueeze(1).clamp(min=0.1)
            )

        # --- 4. Thin center opacity dip (subtle gap hint) ---
        opacity_factor = torch.ones(N, device=device)
        center = norm_dist < 0.1
        if center.any():
            t_c = (norm_dist[center] / 0.1).clamp(0.0, 1.0)
            opacity_factor[center] = 0.5 + 0.5 * t_c
        orig_prob = torch.sigmoid(self._original_opacity)
        new_prob = (orig_prob * opacity_factor.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
        gaussians._opacity.data = torch.log(new_prob / (1.0 - new_prob))

        # --- 5. Red accent at boundary ---
        if self.red_accent > 0:
            red_zone = (norm_dist >= 0.75) & (norm_dist < 1.05)
            red_amount = torch.zeros(N, device=device)
            if red_zone.any():
                t_r = ((norm_dist[red_zone] - 0.9) / 0.15).clamp(-1.0, 1.0)
                red_amount[red_zone] = self.red_accent * (1.0 - t_r * t_r)
            r = red_amount.unsqueeze(1).unsqueeze(1)
            dc = gaussians._features_dc.data
            gaussians._features_dc.data = (1.0 - r) * dc + r * red_sh

    @torch.no_grad()
    def _compute_knn(self, positions: Tensor) -> Tensor:
        """Compute K nearest neighbor indices via chunked cdist (fallback mode)."""
        K = self._knn_k
        N = positions.shape[0]
        knn_idx = torch.zeros(N, K, dtype=torch.long, device=positions.device)

        chunk_size = min(4000, N)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            dists = torch.cdist(positions[i:end], positions)
            _, idx = dists.topk(K + 1, dim=1, largest=False)
            knn_idx[i:end] = idx[:, 1:]

        return knn_idx

    @torch.no_grad()
    def _apply_knn_boundary_viz(self, gaussians, c_surface, x_world):
        """Fallback: KNN-based boundary visualization (before crack paths exist)."""
        c = c_surface.clamp(0.0, 1.0)

        if c.max().item() < 0.01:
            return

        if self._knn_idx is None:
            self._knn_idx = self._compute_knn(x_world)

        c_nbr = c[self._knn_idx]
        c_diff = (c.unsqueeze(1) - c_nbr).abs()
        grad_mag = c_diff.max(dim=1).values

        grad_max = grad_mag.max().clamp(min=0.01)
        grad_norm = (grad_mag / grad_max).clamp(0.0, 1.0)

        boundary = torch.sigmoid(20.0 * (grad_norm - 0.25))
        is_damaged = torch.sigmoid(40.0 * (c - 0.15))
        interior = is_damaged * (1.0 - boundary)

        red_sh = self._red_sh.to(device=c.device).unsqueeze(0)
        dark_sh = self._dark_sh.to(device=c.device).unsqueeze(0)

        eb = boundary.unsqueeze(1).unsqueeze(1)
        ib = interior.unsqueeze(1).unsqueeze(1)
        w_orig = (1.0 - eb - ib).clamp(0.0, 1.0)
        gaussians._features_dc.data = w_orig * self._original_dc + eb * red_sh + ib * dark_sh

        suppress = is_damaged.unsqueeze(1).unsqueeze(1)
        gaussians._features_rest.data = self._original_rest * (1.0 - suppress)

        orig_prob = torch.sigmoid(self._original_opacity)
        opacity_factor = 1.0 - self.crack_opacity_reduction * interior.unsqueeze(1)
        new_prob = (orig_prob * opacity_factor).clamp(1e-6, 1.0 - 1e-6)
        gaussians._opacity.data = torch.log(new_prob / (1.0 - new_prob))

    def update_gaussians(
        self,
        gaussians,
        c_surface: Tensor,
        x_world: Tensor,
        preserve_original: bool = True,
        crack_paths: list = None,
        crack_width: float = 0.03
    ):
        """
        Update Gaussian properties for crack visualization.

        If crack_paths are provided (2+ point polylines), uses physical displacement
        to create visible gaps. Otherwise falls back to KNN boundary coloring.
        """
        # Store originals on first call
        if preserve_original and self._original_dc is None:
            self._original_dc = gaussians._features_dc.data.clone()
            self._original_rest = gaussians._features_rest.data.clone()
            self._original_opacity = gaussians._opacity.data.clone()
            self._original_scaling = gaussians._scaling.data.clone()

        # Set base positions from physics
        gaussians._xyz.data = x_world

        # Reset to originals before applying effects
        gaussians._features_dc.data.copy_(self._original_dc)
        gaussians._features_rest.data.copy_(self._original_rest)
        gaussians._opacity.data.copy_(self._original_opacity)
        gaussians._scaling.data.copy_(self._original_scaling)

        # Route: crack opening (path-based) vs fallback KNN boundary
        has_polylines = (crack_paths is not None
                         and len(crack_paths) > 0
                         and any(p.shape[0] >= 2 for p in crack_paths))

        if has_polylines:
            self._apply_crack_opening(gaussians, crack_paths, crack_width)
        else:
            self._apply_knn_boundary_viz(gaussians, c_surface, x_world)
