"""
Edge-based (bond) phase-field fracture on the Gaussian kNN manifold.

Unlike node-centric phase field (damage on Gaussians), this module tracks
damage on EACH kNN EDGE independently. The framework treats every edge as a
peridynamic-like bond that evolves its own damage variable via AT2-style
phase-field dynamics:

    * Per-edge driving force  psi_edge[i,k] = max(psi[i], psi[knn[i,k]])
    * Per-edge history         H_edge[i,k]   = max(H_edge, psi_edge)   (irreversible)
    * AT2 equilibrium          c_edge_eq = H_ratio / (1 + H_ratio),
                               where H_ratio = 2 * l0 * H_edge / Gc
    * Rate-limited update      c_edge += clamp(c_edge_eq - c_edge, 0, dC_max)
    * Edge break               c_edge > break_threshold → alive_edge = False
    * Broken edges stay broken (c snapped to 1.0, alive flag cannot flip back).

Node damage is then derived as the fraction of broken incident edges,
used for continuous stiffness degradation in the elasticity model.

This formulation is topologically sound — once a crack "line" of edges
breaks, those edges are literally removed from the graph, so connected-
component fragment detection sees a real cut, not a porous obstacle.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from .graph_builder import GaussianGraph


class _CrackFrontStub:
    """Stub that matches the CrackFront interface the simulator reads.

    Edge-based fracture has no crack-tip concept, so tips/visited/parent
    are None. Code that checks `hasattr(ff, 'crack_front')` or reads
    tip_mask will see safe defaults.
    """
    def __init__(self):
        self.tip_mask = None
        self.visited_mask = None
        self.parent_index = None
        self.growth_dir = None
        self.tip_age = None

    def has_active_tips(self) -> bool:
        return False

    def initialize(self, N, device):
        pass

    def get_state(self):
        return {}

    def load_state(self, state):
        pass


class EdgeFractureField:
    """Bond-based phase-field fracture on a Gaussian kNN graph."""

    def __init__(
        self,
        Gc: float = 60000.0,
        l0: float = 0.025,
        dC_max: float = 0.02,
        break_threshold: float = 0.70,
        warmup_frames: int = 3,
        drive_quantile: float = 0.85,
        seed_magnitude: float = 0.35,
        seed_H_multiplier: float = 0.5,
        drive_scale: float = 5.0,
        graph: Optional[GaussianGraph] = None,
        device: str = "cuda",
    ):
        """
        Args:
            Gc: fracture toughness.
            l0: phase-field regularization length.
            dC_max: maximum per-frame damage increment per edge.
            break_threshold: edge is considered broken once c_edge exceeds this.
            warmup_frames: skip damage evolution during early transient frames.
            drive_quantile: robust normalization quantile for psi_edge.
            seed_magnitude: peak c_edge value written by seed_damage().
            seed_H_multiplier: fraction of H_ref added to H_edge by seed_damage().
            graph: reuse a shared GaussianGraph (created if None).
            device: torch device.
        """
        self.Gc = float(Gc)
        self.l0 = float(l0)
        self.dC_max = float(dC_max)
        self.break_threshold = float(break_threshold)
        self.warmup_frames = int(warmup_frames)
        self.drive_quantile = float(drive_quantile)
        self.seed_magnitude = float(seed_magnitude)
        self.seed_H_multiplier = float(seed_H_multiplier)
        self.drive_scale = float(drive_scale)
        self.device = torch.device(device)

        self.graph = graph or GaussianGraph(device=device)

        # Per-edge state (allocated on first initialize/update)
        self.c_edge: Optional[Tensor] = None     # (N, K) damage
        self.H_edge: Optional[Tensor] = None     # (N, K) history
        self.alive_edge: Optional[Tensor] = None # (N, K) bool; True = intact

        # Node-level fields (derived) — exposed as .c, .H for API compatibility
        self.c_node: Optional[Tensor] = None     # (N,) fraction of broken edges
        self.H_node: Optional[Tensor] = None     # (N,) max H across incident edges
        # Legacy compat fields (unused by edge model but read by simulator/visualizer)
        self.a: Optional[Tensor] = None          # (N,) opening magnitude (zeros)
        self.n: Optional[Tensor] = None          # (N, 3) crack normal (zeros)
        self.f: Optional[Tensor] = None          # (N,) fragment label

        # Stub crack_front object so simulator's `.crack_front.tip_mask` queries
        # don't explode when running the edge model.
        self.crack_front = _CrackFrontStub()

        # Metadata
        self.seed_center: Optional[Tensor] = None
        self._frame_count: int = 0
        self._initialized: bool = False
        self._last_diag: Dict[str, float] = {}

        # --- Scheduled-break mode (pattern fracture) ---
        # When attach_pattern() is called, we bypass the physics-driven AT2
        # evolution and break edges on a frame-indexed schedule instead.
        self._pattern_nodes: Optional[Tensor] = None   # (M,) long
        self._pattern_slots: Optional[Tensor] = None   # (M,) long
        self._pattern_frames: Optional[Tensor] = None  # (M,) long (frame offsets)
        self._pattern_base_frame: int = 0              # frame when pattern started
        self._pattern_active: bool = False

    # Legacy API shims — simulator reads .c / .H expecting node-level values.
    @property
    def c(self) -> Optional[Tensor]:
        return self.c_node

    @c.setter
    def c(self, value: Optional[Tensor]) -> None:
        # Allow external writes (e.g., seeding tests) to node-level damage.
        self.c_node = value

    @property
    def H(self) -> Optional[Tensor]:
        return self.H_node

    @H.setter
    def H(self, value: Optional[Tensor]) -> None:
        self.H_node = value

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, positions_or_N) -> None:
        """Allocate state tensors. Accepts positions (N,3) or int N.

        When given int N, caller must build the graph themselves before use.
        When given positions, graph is rebuilt and initial edge lengths are
        recorded for bond-stretch drive computation.
        """
        if isinstance(positions_or_N, int):
            N = positions_or_N
            if self.graph.knn_idx is None:
                self._initialized = False
                return
            K = self.graph.knn_idx.shape[1]
            self._initial_edge_len = None  # populated on first update with positions
        else:
            positions = positions_or_N
            self.graph.build(positions, force=True)
            N, K = self.graph.knn_idx.shape
            # Record edge REST LENGTH (bond-stretch drive later compares
            # current length to this).
            pos_a = positions.unsqueeze(1).expand(-1, K, -1)          # (N, K, 3)
            pos_b = positions[self.graph.knn_idx]                       # (N, K, 3)
            self._initial_edge_len = (pos_b - pos_a).norm(dim=-1).clamp(min=1e-8)
        dev = self.device
        self.c_edge = torch.zeros(N, K, device=dev)
        self.H_edge = torch.zeros(N, K, device=dev)
        self.alive_edge = torch.ones(N, K, dtype=torch.bool, device=dev)
        self.c_node = torch.zeros(N, device=dev)
        self.H_node = torch.zeros(N, device=dev)
        self.a = torch.zeros(N, device=dev)
        self.n = torch.zeros(N, 3, device=dev)
        self.f = torch.zeros(N, dtype=torch.long, device=dev)
        self._frame_count = 0
        self._initialized = True

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        positions: Tensor,
        init_score: Optional[Tensor] = None,
        growth_drive: Optional[Tensor] = None,
        growth_dir: Optional[Tensor] = None,
        F_gaussian: Optional[Tensor] = None,
        impact_center: Optional[Tensor] = None,
        psi_nodes: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Evolve edge damage for one frame.

        Accepts the same kwargs as GaussianFractureField.update() for
        drop-in compatibility. The edge model uses `growth_drive` (or falls
        back to `init_score` or `psi_nodes`) as the per-node driving field.

        Args:
            positions: (N, 3) current Gaussian positions.
            init_score: ignored in edge model (used for tip seeding).
            growth_drive: (N,) physics-derived drive per node — preferred input.
            growth_dir: ignored.
            F_gaussian: ignored.
            impact_center: optional impact center (for seeding / diagnostics).
            psi_nodes: explicit alias for node-level drive.

        Returns:
            dict of per-frame diagnostic stats.
        """
        # Resolve which drive tensor to use (priority: explicit psi_nodes >
        # growth_drive > init_score > zeros).
        if psi_nodes is None:
            if growth_drive is not None:
                psi_nodes = growth_drive
            elif init_score is not None:
                psi_nodes = init_score
            else:
                psi_nodes = torch.zeros(
                    positions.shape[0], device=positions.device, dtype=positions.dtype
                )

        N = positions.shape[0]
        if not self._initialized or self.c_edge is None or self.c_edge.shape[0] != N:
            self.initialize(positions)
        else:
            # Rebuild graph periodically — but ONLY while pattern mode is
            # inactive. Once a pattern is attached, the edge indexing
            # (N, K) is locked to the topology at trigger time; rebuilding
            # would remap neighbors and invalidate alive_edge/c_edge state.
            if not self._pattern_active:
                self.graph.build(positions)

        if impact_center is not None:
            self.seed_center = impact_center.detach().clone()

        self._frame_count += 1
        if self._frame_count <= self.warmup_frames:
            return {"frame": self._frame_count, "warmup": True}

        # --- Pattern mode: scheduled break overrides physics drive ---
        if self._pattern_active:
            # Use absolute frame count for scheduling (simulator's counter).
            # We let the scheduler see whichever frame is currently active.
            current_frame = int(self._last_diag.get("ext_frame", self._frame_count))
            n_new = self._apply_scheduled_breaks(current_frame)
            # Refresh derived node fields.
            broken_count = (~self.alive_edge).sum(dim=1).float()
            K = self.alive_edge.shape[1]
            mean_edge_c = self.c_edge.mean(dim=1)
            broken_frac = broken_count / float(K)
            self.c_node = torch.maximum(mean_edge_c, broken_frac).clamp(0.0, 1.0)
            self.H_node = self.H_edge.max(dim=1).values
            n_broken_total = int((~self.alive_edge).sum().item())
            diag = {
                "frame": self._frame_count,
                "mode": "pattern",
                "n_new_break": n_new,
                "n_broken": n_broken_total,
                "n_total": int(self.alive_edge.numel()),
                "c_node_max": float(self.c_node.max().item()),
            }
            self._last_diag = diag
            if self._frame_count < 5 or n_new > 0:
                print(
                    f"[EdgeFracture:pattern] frame={self._frame_count} "
                    f"new_break={n_new} broken={n_broken_total}/{self.alive_edge.numel()}"
                )
            return diag

        # --- Per-edge driving force ---
        # True peridynamic drive: bond STRETCH (|current_len - rest_len|
        # / rest_len) rather than node-level ψ⁺. This gives a naturally
        # DIRECTIONAL signal: bonds perpendicular to motion stretch, those
        # aligned with motion don't. Under impact, bonds tangent to the
        # contact ring stretch (tension), creating a narrow tensile band
        # that forms a coherent crack line on break — instead of the
        # uniform "whole region damaged" pattern that ψ⁺ drive produces.
        knn = self.graph.knn_idx  # (N, K)
        K = knn.shape[1]
        pos_a = positions.unsqueeze(1).expand(-1, K, -1)     # (N, K, 3)
        pos_b = positions[knn]                                 # (N, K, 3)
        cur_len = (pos_b - pos_a).norm(dim=-1).clamp(min=1e-8) # (N, K)

        if self._initial_edge_len is None or self._initial_edge_len.shape != cur_len.shape:
            # First call with positions — record rest lengths.
            self._initial_edge_len = cur_len.detach().clone()

        rest_len = self._initial_edge_len
        # Signed stretch (positive = tension, negative = compression).
        # Tension-only for fracture (Miehe-style split).
        stretch = (cur_len - rest_len) / rest_len.clamp(min=1e-8)
        stretch_tension = stretch.clamp(min=0.0)

        # Combine stretch drive with a weakened node-ψ signal so we still
        # respond to compressive strain energy in a damped fashion; mostly
        # the stretch dominates.
        psi_a = psi_nodes.unsqueeze(1).expand(-1, K)
        psi_b = psi_nodes[knn]
        psi_pair = torch.minimum(psi_a, psi_b).clamp(min=0.0)

        # Stretch is dimensionless [0, ~1]; scale to H-like units.
        H_ref_raw = self.Gc / (2.0 * self.l0)
        stretch_drive = stretch_tension * H_ref_raw      # each unit stretch ~ H_ref
        psi_drive = psi_pair * 0.25                       # keep ψ as a minor signal
        psi_edge = stretch_drive + psi_drive

        # Robust normalization of drive (prevent outliers from dominating)
        q = float(min(max(self.drive_quantile, 0.5), 0.999))
        if psi_edge.numel() > 0:
            drive_floor = torch.quantile(psi_edge.reshape(-1).detach(), q)
            drive_span = (psi_edge.max() - drive_floor).clamp(min=1e-8)
            psi_norm = ((psi_edge - drive_floor) / drive_span).clamp(0.0, 1.0)
        else:
            psi_norm = psi_edge

        # Scale normalized drive to an H-equivalent magnitude. We use
        # multiple-of-H_ref so the AT2 equilibrium pushes c past the edge
        # break threshold (c_eq = H_ratio / (1 + H_ratio) saturates at 1
        # only as H_ratio → ∞). For break_threshold=0.7 we need H_ratio ≳
        # 2.3, i.e., H_edge ≳ 2.3 * H_ref. Default drive_scale=5 gives
        # plenty of headroom.
        H_ref = self.Gc / (2.0 * self.l0)
        drive_scale = float(getattr(self, "drive_scale", 5.0))
        psi_scaled = psi_norm * H_ref * drive_scale

        # --- History update (irreversible max) ---
        self.H_edge = torch.maximum(self.H_edge, psi_scaled)

        # --- AT2 equilibrium per edge ---
        H_ratio = 2.0 * self.l0 * self.H_edge / self.Gc  # (N, K)
        c_edge_eq = H_ratio / (1.0 + H_ratio)

        # Rate-limited, irreversible update.
        dc = (c_edge_eq - self.c_edge).clamp(0.0, self.dC_max)
        self.c_edge = (self.c_edge + dc).clamp(0.0, 1.0)

        # --- Edge break (snap to 1, disable alive flag) ---
        new_break = (self.c_edge > self.break_threshold) & self.alive_edge
        if bool(new_break.any()):
            self.alive_edge = self.alive_edge & ~new_break
            # Broken edges are c=1 conceptually; keep them saturated.
            self.c_edge = torch.where(
                new_break, torch.ones_like(self.c_edge), self.c_edge
            )

        # Ensure already-broken edges stay saturated (idempotent).
        self.c_edge = torch.where(
            ~self.alive_edge,
            torch.ones_like(self.c_edge),
            self.c_edge,
        )

        # --- Derive node damage from edge state ---
        # Two signals combined: (a) mean edge damage incident to the node,
        # for a smooth degradation signal; (b) fraction of fully-broken
        # edges, for the "really done" regions. Node damage = max of both.
        mean_edge_c = self.c_edge.mean(dim=1)                     # (N,)
        broken_count = (~self.alive_edge).sum(dim=1).float()       # (N,)
        broken_frac = broken_count / float(K)                       # (N,)
        self.c_node = torch.maximum(mean_edge_c, broken_frac).clamp(0.0, 1.0)
        # Track max history per node (for diagnostics / optional physics coupling)
        self.H_node = self.H_edge.max(dim=1).values

        # --- Diagnostics ---
        n_broken = int((~self.alive_edge).sum().item())
        n_total = int(self.alive_edge.numel())
        c_edge_mean = float(self.c_edge.mean().item())
        c_edge_max = float(self.c_edge.max().item())
        c_node_max = float(self.c_node.max().item())
        c_node_mean = float(self.c_node.mean().item())
        n_new = int(new_break.sum().item())
        diag = {
            "frame": self._frame_count,
            "n_broken": n_broken,
            "n_total": n_total,
            "frac_broken": n_broken / max(n_total, 1),
            "n_new_break": n_new,
            "c_edge_max": c_edge_max,
            "c_edge_mean": c_edge_mean,
            "c_node_max": c_node_max,
            "c_node_mean": c_node_mean,
            "H_max": float(self.H_edge.max().item()),
        }
        self._last_diag = diag
        if self._frame_count < 5 or self._frame_count % 10 == 0 or n_new > 0:
            print(
                f"[EdgeFracture] frame={self._frame_count} "
                f"c_edge_max={c_edge_max:.4f} c_node_max={c_node_max:.4f} "
                f"new_break={n_new} broken={n_broken}/{n_total}"
            )
        return diag

    # ------------------------------------------------------------------
    # Pattern (scheduled break) mode
    # ------------------------------------------------------------------
    def set_external_frame(self, frame: int) -> None:
        """Simulator reports its frame counter here so scheduled break
        uses the same timebase as the pattern's break_frames."""
        self._last_diag["ext_frame"] = int(frame)

    @torch.no_grad()
    def attach_pattern(
        self,
        edge_nodes: Tensor,
        edge_slots: Tensor,
        break_frames: Tensor,
        base_frame: int = 0,
    ) -> None:
        """Switch this field into scheduled-break mode using a pre-computed pattern.

        Once attached, `update()` ignores physics drive and simply breaks
        edges whose scheduled frame has arrived. Pattern is applied as an
        overlay: untouched edges remain alive.

        Args:
            edge_nodes: (M,) source node index per pattern edge.
            edge_slots: (M,) kNN slot per pattern edge.
            break_frames: (M,) frame offset (relative to base_frame) when
                edge should break.
            base_frame: simulator frame index at which the schedule starts.
        """
        if not self._initialized:
            raise RuntimeError("EdgeFractureField must be initialized before attach_pattern")
        self._pattern_nodes = edge_nodes.to(self.device).long()
        self._pattern_slots = edge_slots.to(self.device).long()
        self._pattern_frames = break_frames.to(self.device).long()
        self._pattern_base_frame = int(base_frame)
        self._pattern_active = True
        n_edges = int(self._pattern_nodes.numel())
        max_frame = int(self._pattern_frames.max().item()) if n_edges > 0 else 0
        print(
            f"[EdgeFracture] Pattern attached: {n_edges} edges, "
            f"max_break_frame_offset={max_frame}, base_frame={base_frame}"
        )

    @torch.no_grad()
    def _apply_scheduled_breaks(self, current_frame: int) -> int:
        """Break all pattern edges whose scheduled frame has arrived."""
        if not self._pattern_active or self._pattern_nodes is None:
            return 0

        offset = current_frame - self._pattern_base_frame
        due = self._pattern_frames <= offset
        # Only break edges that are still alive (avoid double-counting).
        due_nodes = self._pattern_nodes[due]
        due_slots = self._pattern_slots[due]
        if due_nodes.numel() == 0:
            return 0

        alive_now = self.alive_edge[due_nodes, due_slots]
        newly_due = torch.where(alive_now)[0]
        if newly_due.numel() == 0:
            return 0

        nodes_to_break = due_nodes[newly_due]
        slots_to_break = due_slots[newly_due]

        self.alive_edge[nodes_to_break, slots_to_break] = False
        self.c_edge[nodes_to_break, slots_to_break] = 1.0
        self.H_edge[nodes_to_break, slots_to_break] = torch.maximum(
            self.H_edge[nodes_to_break, slots_to_break],
            torch.full_like(self.H_edge[nodes_to_break, slots_to_break],
                            2.0 * self.Gc / (2.0 * self.l0)),
        )
        return int(nodes_to_break.numel())

    # ------------------------------------------------------------------
    # Impact seeding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def seed_damage(
        self,
        positions: Tensor,
        center: Tensor,
        radius: float,
        magnitude: Optional[float] = None,
        H_multiplier: Optional[float] = None,
    ) -> int:
        """Pre-damage edges whose midpoints lie near an impact center.

        This seeds bond weakening in the contact zone so that subsequent
        physics drive can grow those edges to full break rapidly.
        """
        if not self._initialized:
            self.initialize(positions)

        mag = self.seed_magnitude if magnitude is None else float(magnitude)
        h_mult = self.seed_H_multiplier if H_multiplier is None else float(H_multiplier)

        knn = self.graph.knn_idx
        # Edge midpoint in world space
        pos_a = positions.unsqueeze(1).expand(-1, knn.shape[1], -1)  # (N, K, 3)
        pos_b = positions[knn]                                         # (N, K, 3)
        mid = 0.5 * (pos_a + pos_b)                                    # (N, K, 3)
        dist = (mid - center.unsqueeze(0).unsqueeze(0)).norm(dim=-1)   # (N, K)
        radius = max(float(radius), 1e-6)
        influence = torch.exp(-dist.pow(2) / (2.0 * radius ** 2))       # (N, K)

        seed_c = influence * mag
        self.c_edge = torch.maximum(self.c_edge, seed_c)

        if h_mult > 0.0:
            H_ref = self.Gc / (2.0 * self.l0)
            seed_H = influence * H_ref * h_mult
            self.H_edge = torch.maximum(self.H_edge, seed_H)

        self.seed_center = center.detach().clone()
        # Also pre-damage nodes near impact (max of incident edge seeds)
        # so downstream `c` queries see immediate post-impact damage.
        node_seed = seed_c.max(dim=1).values.clamp(0.0, 0.8)
        if self.c_node is not None:
            self.c_node = torch.maximum(self.c_node, node_seed)
        n_seeded_edges = int((seed_c > 1e-4).sum().item())
        print(
            f"[EdgeFracture] Seeded {n_seeded_edges} edges "
            f"(c_max={float(self.c_edge.max()):.3f}, H_max={float(self.H_edge.max()):.2e})"
        )
        return n_seeded_edges

    # ------------------------------------------------------------------
    # Fragment detection (connected components on alive-edge subgraph)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_fragment_labels(self, min_fragment_size: int = 25) -> Dict[str, object]:
        """Assign connected-component labels using the alive-edge subgraph.

        Undirected edges: i is connected to j if j ∈ kNN(i) and alive_edge[i,slot_j]
        is True. For symmetry, we accept the connection when EITHER direction
        is alive.

        Returns dict with fragment_ids (N,), n_fragments, sizes, and top_sizes.
        """
        if not self._initialized or self.alive_edge is None:
            return {"fragment_ids": None, "n_fragments": 0}

        import numpy as np
        N, K = self.alive_edge.shape
        knn_np = self.graph.knn_idx.cpu().numpy()
        alive_np = self.alive_edge.cpu().numpy()

        # Union-Find
        parent = np.arange(N, dtype=np.int64)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(N):
            for k in range(K):
                if alive_np[i, k]:
                    j = int(knn_np[i, k])
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        labels = np.array([find(i) for i in range(N)], dtype=np.int64)
        unique, counts = np.unique(labels, return_counts=True)
        # Sort by size descending; biggest component = main body (label 0).
        order = np.argsort(-counts)
        sorted_labels = unique[order]
        sorted_counts = counts[order].tolist()

        # Remap labels: 0 = largest, then 1, 2, ... for promoted fragments.
        # Sub-threshold components get merged into label 0 (main body).
        remap = {sorted_labels[0]: 0}
        next_id = 1
        fragment_sizes = [int(sorted_counts[0])]
        for lbl, cnt in zip(sorted_labels[1:], sorted_counts[1:]):
            if cnt >= min_fragment_size:
                remap[int(lbl)] = next_id
                fragment_sizes.append(int(cnt))
                next_id += 1
            else:
                remap[int(lbl)] = 0  # merge into main body
                fragment_sizes[0] += int(cnt)

        remapped = np.array([remap[int(l)] for l in labels], dtype=np.int64)
        fragment_ids = torch.from_numpy(remapped).to(self.device)

        return {
            "fragment_ids": fragment_ids,
            "n_fragments": next_id,
            "fragment_sizes": fragment_sizes,
            "raw_component_count": int(len(unique)),
            "top_component_sizes": sorted_counts[:10],
        }

    # ------------------------------------------------------------------
    # State I/O
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "c_edge": self.c_edge,
            "H_edge": self.H_edge,
            "alive_edge": self.alive_edge,
            "c_node": self.c_node,
            "H_node": self.H_node,
            "seed_center": self.seed_center,
            "frame_count": self._frame_count,
        }

    def load_state(self, state: dict) -> None:
        self.c_edge = state["c_edge"]
        self.H_edge = state["H_edge"]
        self.alive_edge = state["alive_edge"]
        self.c_node = state.get("c_node")
        self.H_node = state.get("H_node")
        self.seed_center = state.get("seed_center")
        self._frame_count = state.get("frame_count", 0)
        self._initialized = self.c_edge is not None
