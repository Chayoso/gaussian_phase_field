"""Run CLIP-driven gravity fracture prompts and save matplotlib visual sequences.

This is a lightweight visualization pass before full Gaussian rendering. It
keeps rendering disabled, runs the gravity-drop manifold simulator, and samples
surface state frames as PNGs so material response and sentence crack style can
be inspected quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.validate_material_sentence_gravity import _slug, _summarize_history
from src.pipeline.manifold_fracture_pipeline import ManifoldFracturePipeline


DEFAULT_PROMPTS = [
    "glass bottle shattering into many sharp radial cracks",
    "glass bottle with one long smooth crack",
    "concrete block crumbling into rough granular chunks",
    "vulcanized rubber ball deforming without visible fracture",
]


def _read_prompts(path: str | None, prompts: list[str] | None) -> list[str]:
    if path:
        prompt_path = Path(path)
        return [
            line.strip().lstrip("\ufeff")
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip().lstrip("\ufeff")
            and not line.lstrip("\ufeff").lstrip().startswith("#")
        ]
    if prompts:
        return list(prompts)
    return list(DEFAULT_PROMPTS)


def _save_visual_config(args, out_dir: Path) -> Path:
    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "particles.target_count", int(args.particles), merge=False)
    OmegaConf.update(cfg, "particles.surface_ratio", 1.0, merge=False)
    OmegaConf.update(cfg, "mpm.num_grids", int(args.grids), merge=False)
    OmegaConf.update(cfg, "rendering.total_frames", int(args.frames), merge=False)
    OmegaConf.update(cfg, "rendering.physics_substeps", int(args.substeps), merge=False)
    OmegaConf.update(cfg, "rendering.render_frames", [], merge=False)
    OmegaConf.update(cfg, "simulation.save_frames", False, merge=False)
    OmegaConf.update(cfg, "simulation.save_checkpoint", False, merge=False)
    OmegaConf.update(cfg, "output.frame_dir", str(out_dir / "frames_unused"), merge=False)
    OmegaConf.update(cfg, "output.checkpoint_dir", str(out_dir / "checkpoints_unused"), merge=False)
    OmegaConf.update(cfg, "output.make_video", False, merge=False)
    OmegaConf.update(cfg, "loading.drop_center_z", float(args.drop_center_z), merge=False)
    OmegaConf.update(cfg, "loading.fixed_gravity_z", float(args.gravity_z), merge=False)
    path = out_dir / "gravity_visual_config.yaml"
    OmegaConf.save(cfg, path)
    return path


def _surface_positions(simulator, source: str = "physical") -> torch.Tensor:
    source = str(source or "physical").lower()
    if source == "physical":
        if getattr(simulator, "x_mpm", None) is not None and getattr(simulator, "surface_mask", None) is not None:
            return simulator.mapper.mpm_to_world(simulator.x_mpm[simulator.surface_mask]).detach()
        gaussians = getattr(simulator, "gaussians", None)
        if gaussians is not None and getattr(gaussians, "_xyz", None) is not None:
            return gaussians._xyz.detach()
        raise RuntimeError("Simulator does not expose physical surface positions")

    render_state = getattr(simulator, "_last_render_state", None) or {}
    render_positions = render_state.get("positions", None)
    if render_positions is not None:
        return render_positions.detach()
    gaussians = getattr(simulator, "gaussians", None)
    if gaussians is not None and getattr(gaussians, "_xyz", None) is not None:
        return gaussians._xyz.detach()
    if getattr(simulator, "x_mpm", None) is not None and getattr(simulator, "surface_mask", None) is not None:
        return simulator.mapper.mpm_to_world(simulator.x_mpm[simulator.surface_mask]).detach()
    raise RuntimeError("Simulator does not expose surface positions")


def _pad_ids(ids: torch.Tensor, n: int) -> torch.Tensor:
    ids = ids.detach()
    if ids.numel() == n:
        return ids
    padded = torch.zeros(n, dtype=ids.dtype, device=ids.device)
    count = min(n, int(ids.numel()))
    if count > 0:
        padded[:count] = ids[:count]
    return padded


def _fragment_ids(simulator, n: int, source: str = "physical") -> torch.Tensor | None:
    source = str(source or "physical").lower()
    if source == "render":
        render_state = getattr(simulator, "_last_render_state", None) or {}
        ids = render_state.get("fragment_ids", None)
        if ids is not None:
            return _pad_ids(ids, n)

    labels = getattr(simulator, "_physical_fragment_labels", None)
    surface_idx = getattr(simulator, "_surface_indices", None)
    if labels is not None and surface_idx is not None and bool((labels > 0).any()):
        if surface_idx.numel() == 0 or int(surface_idx.max().item()) >= int(labels.numel()):
            return None
        ids = labels[surface_idx].detach()
        return _pad_ids(ids, n)

    manager = getattr(simulator, "fragment_manager", None)
    if manager is not None and getattr(manager, "fragment_ids", None) is not None:
        ids = manager.fragment_ids.detach()
        return _pad_ids(ids, n)
    return None


def _plot_state(
    simulator,
    stats: dict,
    out_path: Path,
    title: str,
    max_points: int,
    max_edges: int,
    position_source: str,
) -> None:
    ff = getattr(simulator, "fracture_field", None)
    positions_t = _surface_positions(simulator, source=position_source)
    n = int(positions_t.shape[0])
    if ff is not None and getattr(ff, "c", None) is not None:
        n = min(n, int(ff.c.numel()))
        if getattr(ff, "a", None) is not None:
            n = min(n, int(ff.a.numel()))
        positions_t = positions_t[:n]
        damage_t = ff.c.detach()[:n]
        opening_t = ff.a.detach()[:n] if getattr(ff, "a", None) is not None else torch.zeros(n, device=positions_t.device)
        front = getattr(ff, "crack_front", None)
        visited_t = (
            front.visited_mask.detach()[:n]
            if front is not None and getattr(front, "visited_mask", None) is not None
            else torch.zeros(n, dtype=torch.bool, device=positions_t.device)
        )
        tips_t = (
            front.tip_mask.detach()[:n]
            if front is not None and getattr(front, "tip_mask", None) is not None
            else torch.zeros(n, dtype=torch.bool, device=positions_t.device)
        )
        parent_t = (
            front.parent_index.detach()[:n]
            if front is not None and getattr(front, "parent_index", None) is not None
            else torch.full((n,), -1, dtype=torch.long, device=positions_t.device)
        )
    else:
        positions_t = positions_t[:n]
        damage_t = torch.zeros(n, device=positions_t.device)
        opening_t = torch.zeros(n, device=positions_t.device)
        visited_t = torch.zeros(n, dtype=torch.bool, device=positions_t.device)
        tips_t = torch.zeros(n, dtype=torch.bool, device=positions_t.device)
        parent_t = torch.full((n,), -1, dtype=torch.long, device=positions_t.device)

    frag_t = _fragment_ids(simulator, n, source=position_source)
    if frag_t is not None:
        frag_t = frag_t[:n]

    idx = torch.arange(n, device=positions_t.device)
    if max_points > 0 and n > max_points:
        step = int(np.ceil(n / max_points))
        idx = idx[::step]

    all_positions = positions_t[:n].detach().cpu().numpy()
    all_parent = parent_t.detach().cpu().numpy()
    all_visited = visited_t.detach().cpu().numpy()
    edge_idx = np.where(
        (all_parent >= 0)
        & (all_parent < n)
        & all_visited
        & all_visited[np.clip(all_parent, 0, max(n - 1, 0))]
    )[0]
    if str(position_source).lower() != "physical":
        edge_idx = edge_idx[:0]
    if max_edges > 0 and edge_idx.size > max_edges:
        step = int(np.ceil(edge_idx.size / max_edges))
        edge_idx = edge_idx[::step]

    positions = positions_t[:n][idx].detach().cpu().numpy()
    damage = damage_t[idx].detach().cpu().numpy()
    opening = opening_t[idx].detach().cpu().numpy()
    visited = visited_t[idx].detach().cpu().numpy()
    tips = tips_t[idx].detach().cpu().numpy()
    frag = frag_t[idx].detach().cpu().numpy() if frag_t is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("X-Y top", 0, 1),
        ("X-Z front", 0, 2),
        ("Y-Z side", 1, 2),
    ]
    for ax, (name, i, j) in zip(axes, views):
        if edge_idx.size > 0:
            for child in edge_idx:
                parent = int(all_parent[child])
                ax.plot(
                    [all_positions[parent, i], all_positions[child, i]],
                    [all_positions[parent, j], all_positions[child, j]],
                    color="#0ea5e9",
                    alpha=0.18,
                    linewidth=0.45,
                    zorder=1,
                )
        sc = ax.scatter(
            positions[:, i],
            positions[:, j],
            c=damage,
            cmap="inferno",
            s=0.65,
            alpha=0.72,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
        )
        if frag is not None and np.any(frag > 0):
            active_ids = [frag_id for frag_id in np.unique(frag) if frag_id > 0]
            for frag_id in active_ids:
                mask = frag == frag_id
                ax.scatter(
                    positions[mask, i],
                    positions[mask, j],
                    c=np.full(int(mask.sum()), frag_id),
                    cmap="tab20",
                    s=4.5,
                    alpha=0.92,
                    linewidths=0,
                    vmin=1,
                    vmax=max(len(active_ids), 1),
                )
        if visited.any():
            ax.scatter(positions[visited, i], positions[visited, j], c="#14b8a6", s=2.2, alpha=0.55, linewidths=0)
        if tips.any():
            ax.scatter(positions[tips, i], positions[tips, j], c="#38bdf8", marker="x", s=13.0, alpha=0.95, linewidths=0.7)

        ax.set_title(name)
        ax.set_xlabel("XYZ"[i])
        ax.set_ylabel("XYZ"[j])
        ax.set_aspect("equal")
        margin = 0.04
        ax.set_xlim(float(positions[:, i].min()) - margin, float(positions[:, i].max()) + margin)
        ax.set_ylim(float(positions[:, j].min()) - margin, float(positions[:, j].max()) + margin)

    fig.colorbar(sc, ax=axes, label="damage c", shrink=0.62)
    subtitle = (
        f"frame={stats.get('loop_frame')} impact={bool(stats.get('gravity_contacted', False))} "
        f"frags={stats.get('n_fragments', 1)} cracked={stats.get('n_cracked', 0)} "
        f"cmax={float(stats.get('c_max', damage.max())):.3f} "
        f"open={float(opening.max() if opening.size else 0.0):.4f} "
        f"zmin={float(stats.get('z_min', 0.0)):.3f} zcom={float(stats.get('z_com', 0.0)):.3f} "
        f"src={position_source}"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def _flatten(row: dict) -> dict:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def _write_rows(rows: list[dict], out_dir: Path) -> None:
    json_path = out_dir / "gravity_sentence_visual.json"
    csv_path = out_dir / "gravity_sentence_visual.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten(row))


def _write_report(rows: list[dict], out_dir: Path) -> None:
    lines = [
        "# Gravity Sentence Visual",
        "",
        "Matplotlib sequence run for CLIP-driven gravity fracture. Rendering is disabled; PNGs show surface damage, crack-front visits/tips, and render-safe fragment labels.",
        "",
        "| prompt | family | style | top1 | impact | frags max/final | min render/phys | cracked max/final | cmax | plots | elapsed |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(f"| {row['prompt'][:48]} | ERROR |  |  |  |  |  |  |  |  | {row.get('elapsed_sec', 0):.1f} |")
            continue
        lines.append(
            "| {prompt} | {family} | {style} | {top1} | {impact} | {maxf}/{finalf} | "
            "{min_render}/{min_phys} | {maxc}/{finalc} | {cmax:.3f} | {plots} | {elapsed:.1f} |".format(
                prompt=row["prompt"][:48],
                family=row.get("family", ""),
                style=row.get("sentence_style", ""),
                top1=row.get("top1", ""),
                impact="" if row.get("impact_frame") is None else row.get("impact_frame"),
                maxf=row.get("max_n_fragments", 0),
                finalf=row.get("final_n_fragments", 0),
                min_render=row.get("min_render_fragment_size", 0),
                min_phys=row.get("min_physical_fragment_size", 0),
                maxc=row.get("max_n_cracked", 0),
                finalc=row.get("final_n_cracked", 0),
                cmax=float(row.get("max_c_max", 0.0)),
                plots=len(row.get("plot_frames", [])),
                elapsed=float(row.get("elapsed_sec", 0.0)),
            )
        )
    (out_dir / "gravity_sentence_visual.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(args.prompts_file, args.prompts)
    (out_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    config_path = _save_visual_config(args, out_dir)

    pipeline = ManifoldFracturePipeline(
        str(config_path),
        material_source="clip",
        fast_mode=False,
        db_path=args.db_path,
        clip_model=args.clip_model,
    )

    rows: list[dict] = []
    for idx, prompt in enumerate(prompts):
        prompt_dir = out_dir / f"{idx:02d}_{_slug(prompt)}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "contact_frame": None,
            "scheduled": set(),
            "saved": [],
        }
        t0 = time.time()

        def callback(frame: int, simulator, stats: dict) -> None:
            contacted = bool(stats.get("gravity_contacted", False))
            if contacted and state["contact_frame"] is None:
                state["contact_frame"] = int(frame)
                offsets = [0, 1, 2, 4, 8, 12, 16, 24]
                state["scheduled"].update(int(frame) + offset for offset in offsets)

            should_save = (
                frame == 0
                or frame == int(args.frames) - 1
                or frame % max(int(args.plot_every), 1) == 0
                or int(frame) in state["scheduled"]
            )
            if not should_save:
                return

            out_path = prompt_dir / f"state_{frame:04d}.png"
            _plot_state(
                simulator=simulator,
                stats=stats,
                out_path=out_path,
                title=f"{prompt}",
                max_points=int(args.max_plot_points),
                max_edges=int(args.max_plot_edges),
                position_source=str(args.position_source),
            )
            state["saved"].append(str(out_path))

        print(f"[visual:gravity] {idx + 1}/{len(prompts)} {prompt!r}", flush=True)
        try:
            result = pipeline.run(
                prompt,
                num_frames=int(args.frames),
                save_frames=False,
                return_frames=False,
                override_params={
                    "rendering.render_frames": [],
                    "output.make_video": False,
                    "simulation.save_checkpoint": False,
                },
                state_callback=callback,
            )
            history = list(getattr(pipeline.engine, "last_stats_history", []))
            prior = result["material_prior"]
            row = {
                "prompt": prompt,
                "family": result["fracture_family"],
                "sentence_style": prior.get("sentence_style", "material_default"),
                "dominant_category": result["material_category"],
                "top1": prior["top_k"][0]["name"] if prior["top_k"] else "",
                "top_k": prior["top_k"],
                "E_mpm": result["params"]["E"],
                "Gc_mpm": result["params"]["Gc"],
                "nu_mpm": result["params"]["nu"],
                "density_mpm": result["params"]["density"],
                "elapsed_sec": time.time() - t0,
                "plot_dir": str(prompt_dir),
                "plot_frames": state["saved"],
            }
            row.update(_summarize_history(history))
        except Exception as exc:
            row = {
                "prompt": prompt,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": time.time() - t0,
                "plot_dir": str(prompt_dir),
                "plot_frames": state["saved"],
            }
            print(f"[visual:gravity:error] {prompt!r}: {exc}", flush=True)
        rows.append(row)
        _write_rows(rows, out_dir)
        _write_report(rows, out_dir)
        if row.get("error") and "CUDA error" in str(row.get("error")):
            raise SystemExit(1)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    _write_rows(rows, out_dir)
    _write_report(rows, out_dir)
    print(f"Gravity visual run complete: {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP sentence/material gravity fracture visual sequence.")
    parser.add_argument("--config", default="configs/gravity_drop_manifold.yaml")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--out", default="output/gravity_sentence_visual")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--particles", type=int, default=10000)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--grids", type=int, default=64)
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--drop-center-z", type=float, default=0.42)
    parser.add_argument("--gravity-z", type=float, default=-3500.0)
    parser.add_argument("--plot-every", type=int, default=8)
    parser.add_argument("--max-plot-points", type=int, default=50000)
    parser.add_argument("--max-plot-edges", type=int, default=6000)
    parser.add_argument("--position-source", choices=["physical", "render"], default="physical")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
