"""
Physics-conditioned procedural fracture pattern synthesizer.

Given a FractureTrigger (nucleation site + direction + stress magnitude) and
a material descriptor (E, Gc, ν, ρ — from config or CLIP predictor), this
module generates a fracture pattern as:
    * a set of crack edges in the kNN graph (indices into (N, K) edge grid)
    * a per-edge activation schedule (frame offset until each edge breaks)

Supported patterns:
    * "radial_rings"  — N radial spokes from nucleation + M concentric
                       geodesic rings. Good for impact shatter.
    * "voronoi"       — Poisson-disk seeds biased toward nucleation,
                       Voronoi edges extracted from kNN graph. Good for
                       generic brittle shatter.
    * "none"          — no pattern; for ductile or soft materials.

Material → pattern parameters mapping is encapsulated in
`material_to_pattern_spec()`. CLIP-predicted {E, Gc, ν, ρ} plugs in here
naturally — higher brittleness index (√(E / Gc)) → denser pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import torch
from torch import Tensor

from .physics_trigger import FractureTrigger
from .graph_builder import GaussianGraph


# ----------------------------------------------------------------------
# Pattern spec: parameters that control pattern generation.
# Decoupled from the material so CLIP-based predictors can produce one.
# ----------------------------------------------------------------------
@dataclass
class PatternSpec:
    """Procedural pattern configuration (material-independent once derived)."""
    pattern_type: str = "radial_rings"   # "radial_rings" | "voronoi" | "none"
    n_spokes: int = 8                    # radial_rings only
    n_rings: int = 3                     # radial_rings only
    ring_spacing_hops: int = 8           # BFS depth between rings
    voronoi_n_cells: int = 40            # voronoi only
    voronoi_center_bias: float = 0.6     # 0=uniform, 1=all at nucleation
    propagation_hops_per_frame: float = 2.0  # crack front speed (BFS hops/frame)
    pattern_radius_hops: int = 40        # max BFS reach for the pattern
    halo_hops: int = 1                   # edge thickening (0=no halo, 1=1-ring)


# ----------------------------------------------------------------------
# Material → PatternSpec mapping.
# CLIP-predicted material parameters feed directly into this.
# ----------------------------------------------------------------------
def material_to_pattern_spec(
    E: float,
    Gc: float,
    nu: float = 0.25,
    density: float = 1200.0,
    stress_magnitude: Optional[float] = None,
    *,
    base_density: int = 30,
    # Reference material for scaling (tuned to produce ~30 fragments on bunny):
    E_ref: float = 1.5e7,
    Gc_ref: float = 60000.0,
    stress_ref: float = 1.0e5,       # reference σ₁ magnitude (typical brittle impact)
    pattern_type_override: Optional[str] = None,
) -> PatternSpec:
    """Derive a PatternSpec from physical material parameters.

    Intuition:
      * Brittleness index B = √(E / Gc). High B → brittle → many small pieces.
      * Pattern type:
          - B very high (glass/ceramic): voronoi, dense
          - B medium (concrete/stone):   radial_rings
          - B low (wood/polymer):        voronoi, sparse
          - Extremely low (rubber/metal): "none"
      * Propagation speed ∝ √(E / ρ) — elastic wave speed.
    """
    B = math.sqrt(max(E / max(Gc, 1e-9), 1e-9))
    B_ref = math.sqrt(E_ref / max(Gc_ref, 1e-9))
    brittleness = B / max(B_ref, 1e-9)  # normalized, 1.0 = reference

    # --- Pattern type selection ---
    if pattern_type_override is not None:
        ptype = pattern_type_override
    elif brittleness > 1.5:
        ptype = "voronoi"
    elif brittleness > 0.6:
        ptype = "radial_rings"
    elif brittleness > 0.2:
        ptype = "voronoi"  # sparse, used for ductile-ish
    else:
        ptype = "none"

    # --- Pattern density ---
    # Scale by brittleness (material) AND stress magnitude (impact energy).
    # Physics argument: higher impact energy → more energy available for
    # new crack surfaces → more fragments. The scaling exponent is mild
    # (0.3) so a 10x impact energy roughly doubles fragment count.
    if stress_magnitude is not None and stress_magnitude > 0:
        stress_factor = (stress_magnitude / max(stress_ref, 1e-9)) ** 0.3
        stress_factor = max(0.3, min(stress_factor, 4.0))
    else:
        stress_factor = 1.0

    # Scale factor — shared base so config's base_density actually controls
    # fragment count for radial_rings pattern too (not just voronoi).
    density_mul = max(1.0, base_density / 30.0)
    n_cells = max(4, int(round(base_density * brittleness ** 1.2 * stress_factor)))
    n_spokes = max(4, int(round(16 * density_mul * brittleness ** 0.8 * stress_factor ** 0.7)))
    # Rings scale sublinearly with density_mul so total edge count stays
    # bounded: pattern_edges ≈ n_spokes * (BFS_reach / ring_spacing) so pushing
    # both to 10+ multiplies edge count beyond MPM stability.
    n_rings = max(1, int(round(3 * (density_mul ** 0.5) * brittleness ** 0.5 * stress_factor ** 0.5)))

    # --- Propagation speed ---
    # Wave speed c ≈ √(E/ρ). Map to ~2 hops/frame at reference.
    wave_speed = math.sqrt(max(E, 1e-3) / max(density, 1e-3))
    wave_speed_ref = math.sqrt(E_ref / max(density, 1e-3))
    prop_hops = 2.0 * (wave_speed / max(wave_speed_ref, 1e-6))
    prop_hops = max(0.5, min(prop_hops, 6.0))

    # --- Ring spacing: scale with Gc (tougher material → wider-spaced cracks) ---
    ring_spacing = max(3, int(round(8.0 * (Gc / max(Gc_ref, 1e-9)) ** 0.5)))

    # --- Pattern reach ---
    # Full-shatter mode: pattern covers the entire reachable surface so
    # extremities (ears, tail, head) also crack. A typical bunny has BFS
    # diameter ~90 hops; we set 200 as a generous upper bound so the BFS
    # terminates by exhaustion, not by max_depth.
    pattern_radius = max(80, int(round(200 * brittleness ** 0.25 * stress_factor ** 0.5)))

    return PatternSpec(
        pattern_type=ptype,
        n_spokes=n_spokes,
        n_rings=n_rings,
        ring_spacing_hops=ring_spacing,
        voronoi_n_cells=n_cells,
        voronoi_center_bias=0.6,
        propagation_hops_per_frame=prop_hops,
        pattern_radius_hops=pattern_radius,
        halo_hops=1,
    )


# ----------------------------------------------------------------------
# Synthesized pattern: concrete edge list + schedule.
# ----------------------------------------------------------------------
@dataclass
class SynthesizedPattern:
    """Concrete output of pattern generation.

    Edge indexing: each entry is (node_i, slot_k) where slot_k indexes into
    graph.knn_idx[node_i, :]. The edge connects node_i → graph.knn_idx[i, k].
    """
    edge_nodes: Tensor              # (M,) long — source node for each pattern edge
    edge_slots: Tensor              # (M,) long — kNN slot for each pattern edge
    break_frames: Tensor            # (M,) long — frame offset when edge breaks
    nucleation_idx: int             # For reference/debug
    pattern_type: str
    # Optional diagnostics:
    info: Dict[str, int] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Main synthesizer class
# ----------------------------------------------------------------------
class PatternSynthesizer:
    """Generates a procedural fracture pattern conditioned on physics + material."""

    def __init__(self, graph: GaussianGraph, device: str = "cuda"):
        self.graph = graph
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        positions: Tensor,
        trigger: FractureTrigger,
        spec: PatternSpec,
        stress_per_node: Optional[Tensor] = None,
    ) -> Optional[SynthesizedPattern]:
        """Generate a crack pattern around the trigger.

        Args:
            positions: (N, 3) surface Gaussian positions.
            trigger: Physics trigger (nucleation site + direction).
            spec: Pattern parameters.
            stress_per_node: Optional (N,) σ₁ magnitude per Gaussian. When
                provided, BFS is stress-weighted so high-stress regions
                effectively shrink and accumulate more rings (denser pattern
                where physics concentrates stress).
        """
        if spec.pattern_type == "none" or self.graph.knn_idx is None:
            return None

        # BFS from nucleation to assign per-node graph distance.
        # Use UNFILTERED kNN so the pattern can reach extremities like
        # ears/tail even when the normal filter disconnects front and
        # back shells. (Fragment CC downstream still uses the filtered
        # alive_edge to preserve surface topology.)
        depth = self._bfs_depth(
            self.graph.knn_idx,
            trigger.nucleation_idx,
            max_depth=spec.pattern_radius_hops,
            weights=None,  # use raw kNN connectivity for pattern reach
            stress_per_node=stress_per_node,
        )
        reach_count = int((depth >= 0).sum().item())
        print(f"[PatternSynth] BFS reach = {reach_count}/{depth.numel()} "
              f"(max_depth={int(depth[depth>=0].max().item()) if reach_count > 0 else 0})")

        if spec.pattern_type == "radial_rings":
            pattern = self._pattern_radial_rings(
                positions, trigger, spec, depth
            )
        elif spec.pattern_type == "voronoi":
            pattern = self._pattern_voronoi(positions, trigger, spec, depth)
        else:
            return None

        if pattern is None:
            return None

        # Halo thickening: for each pattern edge, also schedule nearby edges
        # so long-range kNN links can't bypass the thin pattern. Iterate
        # halo_hops times to thicken by multiple rings.
        for _ in range(max(0, spec.halo_hops)):
            pattern = self._thicken_pattern(pattern, 1)

        return pattern

    # ------------------------------------------------------------------
    # Pattern: Radial + rings
    # ------------------------------------------------------------------
    def _pattern_radial_rings(
        self,
        positions: Tensor,
        trigger: FractureTrigger,
        spec: PatternSpec,
        depth: Tensor,
    ) -> Optional[SynthesizedPattern]:
        """Radial spokes from nucleation + concentric geodesic rings.

        Edges on a spoke share the 'radial' property: both endpoints sit
        along one of N azimuthal corridors. Edges on a ring share the
        'tangential' property: both endpoints lie at the same BFS depth.
        Break schedule propagates outward: inner edges break first.
        """
        N = positions.shape[0]
        knn = self.graph.knn_idx
        K = knn.shape[1]
        reachable = depth >= 0
        if not bool(reachable.any()):
            return None

        # --- STRESS-ALIGNED azimuth ---
        # Physics argument: cracks form PERPENDICULAR to the principal tensile
        # direction σ₁ (the material pulls apart along σ₁; crack surfaces
        # open along ±σ₁). So spoke corridors should be measured in the
        # plane PERPENDICULAR to σ₁ at the nucleation point. This makes
        # the pattern's orientation a direct physics output, not a
        # geometric choice.
        nuc_pos = trigger.nucleation_pos.to(positions.device)
        principal = trigger.principal_dir.to(positions.device)
        principal = torch.nn.functional.normalize(principal, dim=-1)

        # Build orthonormal frame (e1, e2) spanning the plane ⊥ to σ₁.
        # e1: any vector ⊥ to σ₁ (prefer global up or x).
        if principal.norm() < 1e-8:
            # Degenerate case: fall back to XY.
            e1 = torch.tensor([1.0, 0.0, 0.0], device=principal.device,
                              dtype=principal.dtype)
            e2 = torch.tensor([0.0, 1.0, 0.0], device=principal.device,
                              dtype=principal.dtype)
        else:
            ref = (
                torch.tensor([0.0, 0.0, 1.0], device=principal.device,
                             dtype=principal.dtype)
                if principal[2].abs() < 0.9
                else torch.tensor([1.0, 0.0, 0.0], device=principal.device,
                                  dtype=principal.dtype)
            )
            e1 = ref - (ref @ principal) * principal
            e1 = torch.nn.functional.normalize(e1, dim=-1)
            e2 = torch.cross(principal, e1, dim=0)
            e2 = torch.nn.functional.normalize(e2, dim=-1)

        # Project each node's displacement onto the (e1, e2) plane.
        d_pos = positions - nuc_pos.unsqueeze(0)  # (N, 3)
        u = d_pos @ e1                              # (N,)
        v = d_pos @ e2                              # (N,)
        phi = torch.atan2(v, u)                     # (N,) [-π, π] in stress-plane

        # Assign each reachable node to an angular sector (0..n_spokes-1).
        n_spokes = max(spec.n_spokes, 1)
        two_pi = 2.0 * math.pi
        sector = ((phi + math.pi) / two_pi * n_spokes).long().clamp(0, n_spokes - 1)
        sector_width = two_pi / n_spokes
        sector_center = -math.pi + (sector.float() + 0.5) * sector_width
        phi_err = (phi - sector_center).abs()
        phi_err = torch.minimum(phi_err, two_pi - phi_err)

        # Ring assignments by BFS depth. Distribute rings EVENLY across the
        # full reached BFS range (not just ring_spacing_hops fixed intervals)
        # so the pattern covers the whole surface rather than a narrow band
        # near the nucleation.
        max_depth_reached = int(depth[reachable].max().item()) if bool(reachable.any()) else 0
        n_rings = max(spec.n_rings, 1)
        if max_depth_reached > 0:
            # Divide by n_rings (not n_rings+1) so the outermost ring sits at
            # ~max_depth, guaranteeing a ring cut through extremities (ears,
            # tail, head tips) rather than leaving a gap between the last
            # ring and the BFS frontier.
            ring_spacing = max(2, int(max_depth_reached / n_rings))
        else:
            ring_spacing = max(spec.ring_spacing_hops, 2)
        ring_id = torch.div(depth.clamp(min=0), ring_spacing, rounding_mode="floor")
        # Only depths that fall within a ring_half_band are ring-eligible.
        ring_half_band = 1
        depth_mod = (depth - ring_id * ring_spacing).abs()
        is_ring_depth = (
            reachable
            & (depth_mod <= ring_half_band)
            & (ring_id >= 1)                  # skip ring at depth 0 (nucleation)
            & (ring_id <= n_rings)
        )

        # For each pattern edge, need both endpoints reachable AND the edge
        # must be either SPOKE-like (both endpoints in same angular sector
        # with small phi_err) or RING-like (both endpoints at the same ring
        # depth).
        # Expand to edge grid.
        i_idx = torch.arange(N, device=positions.device).unsqueeze(1).expand(-1, K)
        j_idx = knn
        dep_i = depth.unsqueeze(1).expand(-1, K)
        dep_j = depth[j_idx]
        reach_i = (dep_i >= 0)
        reach_j = (dep_j >= 0)
        both_reach = reach_i & reach_j

        # Only use surface-aware edges (non-zero weight after normal filter).
        # Otherwise the pattern edges include through-shell shortcuts that
        # don't participate in the surface-manifold topology.
        if self.graph.weights is not None:
            w_edge = self.graph.weights > 1e-8  # (N, K)
        else:
            w_edge = torch.ones_like(j_idx, dtype=torch.bool)
        both_reach = both_reach & w_edge

        # Spoke edge: CROSSES a sector boundary (different sector ids).
        # These are the edges we want to break so adjacent pie-slices
        # separate. Edges INSIDE a sector are kept alive (they hold the
        # pie-slice together as a fragment).
        sector_i = sector.unsqueeze(1).expand(-1, K)
        sector_j = sector[j_idx]
        spoke_edge = (sector_i != sector_j) & both_reach

        # Ring edge: CROSSES a ring boundary (different ring ids), with at
        # least one endpoint actually sitting at the ring depth band.
        # These are the circumferential cracks that separate nested annuli.
        ring_i = ring_id.unsqueeze(1).expand(-1, K)
        ring_j = ring_id[j_idx]
        ring_mask_i = is_ring_depth.unsqueeze(1).expand(-1, K)
        ring_mask_j = is_ring_depth[j_idx]
        ring_edge = (
            (ring_i != ring_j)
            & (ring_mask_i | ring_mask_j)
            & both_reach
        )

        pattern_mask = spoke_edge | ring_edge
        pattern_idx = torch.where(pattern_mask.reshape(-1))[0]
        if pattern_idx.numel() == 0:
            return None

        # Recover (node, slot) from flat index.
        edge_nodes = pattern_idx // K
        edge_slots = pattern_idx % K

        # Break schedule: propagation outward from nucleation.
        max_depth_for_edge = torch.maximum(dep_i, dep_j).reshape(-1)
        edge_max_depth = max_depth_for_edge[pattern_idx]
        speed = max(spec.propagation_hops_per_frame, 0.5)
        break_frames = (edge_max_depth.float() / speed).long().clamp(min=0)

        return SynthesizedPattern(
            edge_nodes=edge_nodes.to(self.device),
            edge_slots=edge_slots.to(self.device),
            break_frames=break_frames.to(self.device),
            nucleation_idx=trigger.nucleation_idx,
            pattern_type="radial_rings",
            info={
                "n_edges": int(pattern_idx.numel()),
                "n_spoke_edges": int(spoke_edge.sum().item()),
                "n_ring_edges": int(ring_edge.sum().item()),
                "max_depth": int(max_depth_for_edge.max().item()),
            },
        )

    # ------------------------------------------------------------------
    # Pattern: Voronoi
    # ------------------------------------------------------------------
    def _pattern_voronoi(
        self,
        positions: Tensor,
        trigger: FractureTrigger,
        spec: PatternSpec,
        depth: Tensor,
    ) -> Optional[SynthesizedPattern]:
        """Voronoi tessellation with seeds biased toward the nucleation point.

        Seed placement:
          * Half of the cells seeded by Poisson-disk sampling near the
            nucleation (radius controlled by voronoi_center_bias).
          * Remaining seeded uniformly across reachable nodes.

        Voronoi edges: edges in the kNN graph whose endpoints belong to
        DIFFERENT Voronoi cells.
        """
        N = positions.shape[0]
        knn = self.graph.knn_idx
        K = knn.shape[1]

        reachable = depth >= 0
        reachable_idx = torch.where(reachable)[0]
        if reachable_idx.numel() < spec.voronoi_n_cells:
            return None

        n_cells = int(max(spec.voronoi_n_cells, 2))
        # Biased seed set: draw nearer-to-nucleation nodes preferentially.
        nuc_pos = trigger.nucleation_pos.to(positions.device)
        dist_to_nuc = (positions[reachable_idx] - nuc_pos.unsqueeze(0)).norm(dim=1)
        # Weight ∝ exp(-bias * dist / median_dist). Higher bias → tighter.
        bias = float(max(spec.voronoi_center_bias, 0.0))
        if bias > 0.0:
            median = dist_to_nuc.median().clamp(min=1e-6)
            weights = torch.exp(-bias * dist_to_nuc / median)
        else:
            weights = torch.ones_like(dist_to_nuc)
        weights = weights / weights.sum()

        # Sample seed indices WITHOUT replacement, roughly Poisson-disk-ish.
        # For simplicity: draw 3x candidates, pick greedily by nearest-seed spacing.
        n_candidates = min(reachable_idx.numel(), n_cells * 4)
        candidate_perm = torch.multinomial(weights, n_candidates, replacement=False)
        candidate_nodes = reachable_idx[candidate_perm]

        selected: List[int] = []
        sel_pos_list: List[Tensor] = []
        min_spacing = float(
            (positions[reachable_idx].max(dim=0).values
             - positions[reachable_idx].min(dim=0).values).norm().item()
        ) / max(math.sqrt(n_cells), 1.0)
        min_spacing *= 0.4  # allow some overlap
        for idx in candidate_nodes.tolist():
            p = positions[idx]
            ok = True
            for sp in sel_pos_list:
                if (p - sp).norm().item() < min_spacing:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                sel_pos_list.append(p)
            if len(selected) >= n_cells:
                break

        if len(selected) < 2:
            return None

        seed_pos = torch.stack(sel_pos_list, dim=0)  # (S, 3)
        # Assign each node to the nearest seed (Voronoi cell labels).
        # For scale, chunk through cdist.
        chunk = 8192
        cell = torch.zeros(N, dtype=torch.long, device=positions.device)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            d = torch.cdist(positions[start:end], seed_pos)
            cell[start:end] = d.argmin(dim=1)

        # Voronoi edge: endpoints in different cells.
        cell_i = cell.unsqueeze(1).expand(-1, K)
        cell_j = cell[knn]
        diff_cell = cell_i != cell_j
        # Require both endpoints reachable (pattern only fires in the impact region).
        dep_i = depth.unsqueeze(1).expand(-1, K)
        dep_j = depth[knn]
        reach_edge = (dep_i >= 0) & (dep_j >= 0)
        pattern_mask = diff_cell & reach_edge

        pattern_idx = torch.where(pattern_mask.reshape(-1))[0]
        if pattern_idx.numel() == 0:
            return None

        edge_nodes = pattern_idx // K
        edge_slots = pattern_idx % K

        # Break schedule: propagation by max-endpoint-depth.
        max_depth_for_edge = torch.maximum(dep_i, dep_j).reshape(-1)
        edge_max_depth = max_depth_for_edge[pattern_idx]
        speed = max(spec.propagation_hops_per_frame, 0.5)
        break_frames = (edge_max_depth.float() / speed).long().clamp(min=0)

        return SynthesizedPattern(
            edge_nodes=edge_nodes.to(self.device),
            edge_slots=edge_slots.to(self.device),
            break_frames=break_frames.to(self.device),
            nucleation_idx=trigger.nucleation_idx,
            pattern_type="voronoi",
            info={
                "n_edges": int(pattern_idx.numel()),
                "n_cells": len(selected),
                "max_depth": int(max_depth_for_edge.max().item()),
            },
        )

    # ------------------------------------------------------------------
    # Halo thickening: add 1-hop edges around each pattern edge.
    # ------------------------------------------------------------------
    def _thicken_pattern(
        self,
        pattern: SynthesizedPattern,
        halo_hops: int,
    ) -> SynthesizedPattern:
        """Expand each pattern edge by 1-hop halo so kNN long-range edges
        can't bypass the thin pattern barrier."""
        knn = self.graph.knn_idx
        K = knn.shape[1]
        N = knn.shape[0]

        # Collect all nodes touched by the pattern (both endpoints).
        touched_nodes = torch.cat(
            [pattern.edge_nodes, knn[pattern.edge_nodes, pattern.edge_slots]]
        )
        touched_nodes = torch.unique(touched_nodes)

        # Edges whose SOURCE is a touched node → added as halo (scheduled
        # to break at the same frame as the originating pattern edge).
        # Build a node→earliest_break_frame lookup.
        first_break = torch.full(
            (N,), 10_000_000, dtype=torch.long, device=pattern.break_frames.device
        )
        # For each pattern edge, the source node's earliest break frame.
        src_nodes = pattern.edge_nodes
        frames = pattern.break_frames
        first_break.scatter_reduce_(
            0, src_nodes, frames, reduce="amin", include_self=True
        )
        # Also the destination nodes — so halo around destination works.
        dst_nodes = knn[pattern.edge_nodes, pattern.edge_slots]
        first_break.scatter_reduce_(
            0, dst_nodes, frames, reduce="amin", include_self=True
        )

        # Gather edges whose source is touched.
        src_mask = torch.zeros(N, dtype=torch.bool, device=pattern.break_frames.device)
        src_mask[touched_nodes] = True

        # All edges emanating from touched nodes.
        all_nodes = torch.arange(N, device=pattern.break_frames.device)
        src_emanating = all_nodes.unsqueeze(1).expand(-1, K)  # (N, K)
        src_mask_edge = src_mask.unsqueeze(1).expand(-1, K)
        halo_mask = src_mask_edge.reshape(-1)
        halo_idx = torch.where(halo_mask)[0]

        halo_nodes = halo_idx // K
        halo_slots = halo_idx % K

        # Assign break time = source node's first break frame + 1 (slightly
        # after the core edge).
        halo_frames = (first_break[halo_nodes] + 1).clamp(min=0)

        # Merge with original pattern (dedup by (node, slot)).
        all_nodes_cat = torch.cat([pattern.edge_nodes, halo_nodes])
        all_slots_cat = torch.cat([pattern.edge_slots, halo_slots])
        all_frames_cat = torch.cat([pattern.break_frames, halo_frames])

        # Flatten (node, slot) pairs to unique keys.
        keys = all_nodes_cat * K + all_slots_cat
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        # Take the MINIMUM break frame for duplicates.
        merged_frames = torch.full(
            unique_keys.shape, 10_000_000, dtype=torch.long,
            device=all_frames_cat.device,
        )
        merged_frames.scatter_reduce_(
            0, inverse, all_frames_cat, reduce="amin", include_self=True
        )
        out_nodes = unique_keys // K
        out_slots = unique_keys % K

        n_halo = int(halo_idx.numel())
        return SynthesizedPattern(
            edge_nodes=out_nodes,
            edge_slots=out_slots,
            break_frames=merged_frames,
            nucleation_idx=pattern.nucleation_idx,
            pattern_type=pattern.pattern_type,
            info={**pattern.info, "n_halo": n_halo, "n_merged_edges": int(unique_keys.numel())},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _bfs_depth(
        knn_idx: Tensor,
        source: int,
        max_depth: int = 40,
        weights: Optional[Tensor] = None,
        weight_threshold: float = 1e-8,
        stress_per_node: Optional[Tensor] = None,
    ) -> Tensor:
        """Surface-aware (optionally stress-weighted) geodesic distance on kNN.

        * If `weights` is provided, edges with weight ≤ threshold are skipped
          (normal-filtered edges through thin shells).
        * If `stress_per_node` is provided, each hop contributes an effective
          cost scaled by (1 / (1 + α σ₁)), where α is tuned so that high-
          stress regions have SHORTER effective distance. The returned
          "depth" is then QUANTIZED into integer BFS levels so downstream
          ring assignment still works. Physically: rings cluster densely
          where stress is high (more fragments in impact zone).

        Returns (N,) long with -1 for unreachable nodes.
        """
        N, K = knn_idx.shape
        device = knn_idx.device
        import numpy as np

        if stress_per_node is None:
            # Classic uniform BFS (level-set by hop count).
            depth = torch.full((N,), -1, dtype=torch.long, device=device)
            depth[source] = 0
            frontier = [int(source)]
            knn_cpu = knn_idx.cpu().numpy()
            w_cpu = weights.cpu().numpy() if weights is not None else None
            depth_np = depth.cpu().numpy()
            for d in range(max_depth):
                if not frontier:
                    break
                next_front = []
                for u in frontier:
                    for k in range(K):
                        if w_cpu is not None and w_cpu[u, k] <= weight_threshold:
                            continue
                        v = int(knn_cpu[u, k])
                        if depth_np[v] < 0:
                            depth_np[v] = d + 1
                            next_front.append(v)
                frontier = next_front
            return torch.from_numpy(depth_np).to(device)

        # Stress-weighted Dijkstra on the kNN graph.
        # Normalize stress to roughly [0, 1]: divide by the 95th percentile.
        s = stress_per_node.clamp(min=0.0)
        s_ref = torch.quantile(s, 0.95).clamp(min=1e-8)
        s_norm = (s / s_ref).clamp(0.0, 4.0)
        # Effective edge cost: 1 / (1 + alpha * min(s_i, s_j)).
        # High stress on both ends → short effective cost.
        alpha = 3.0
        # Precompute node weights on CPU.
        s_np = s_norm.cpu().numpy()
        knn_cpu = knn_idx.cpu().numpy()
        w_cpu = weights.cpu().numpy() if weights is not None else None

        cost = np.full(N, np.inf, dtype=np.float64)
        cost[source] = 0.0
        # Priority queue.
        import heapq
        pq = [(0.0, int(source))]
        max_cost_budget = float(max_depth)  # reuse max_depth as cost cap
        while pq:
            cu, u = heapq.heappop(pq)
            if cu > cost[u] + 1e-12:
                continue
            if cu > max_cost_budget:
                continue
            for k in range(K):
                if w_cpu is not None and w_cpu[u, k] <= weight_threshold:
                    continue
                v = int(knn_cpu[u, k])
                step = 1.0 / (1.0 + alpha * min(s_np[u], s_np[v]))
                cv = cu + step
                if cv < cost[v]:
                    cost[v] = cv
                    if cv <= max_cost_budget:
                        heapq.heappush(pq, (cv, v))

        # Quantize the continuous cost to integer depth levels (granularity
        # 0.5 → 2x finer than hop-level). This lets downstream ring code
        # treat "depth" uniformly.
        granularity = 0.5
        depth_np = np.where(np.isinf(cost), -1, (cost / granularity).astype(np.int64))
        return torch.from_numpy(depth_np).to(device)
