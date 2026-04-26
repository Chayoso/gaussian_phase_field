"""Build a compact visual/metric report for validation runs."""

from __future__ import annotations

import argparse
import csv
from math import ceil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


METRIC_COLUMNS = (
    "sentence_style",
    "family",
    "top1",
    "max_n_fragments",
    "max_n_cracked",
    "final_visited_count",
    "final_crack_hoop_fraction",
    "final_crack_nonradial_fraction",
    "final_visited_angular_entropy",
    "max_c_max",
    "elapsed_sec",
)


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _short(text: str, limit: int = 58) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _resolve_plot(row: dict, run_dir: Path, prefix: str, idx: int) -> Path | None:
    explicit = row.get("final_plot")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path

    plot_dir = run_dir / "final_plots" / prefix
    matches = sorted(plot_dir.glob(f"{idx:02d}_*.png"))
    if matches:
        return matches[0]
    return None


def write_contact_sheet(rows: list[dict], run_dir: Path, prefix: str, out_path: Path) -> None:
    resolved = [(row, _resolve_plot(row, run_dir, prefix, idx)) for idx, row in enumerate(rows)]
    resolved = [(row, path) for row, path in resolved if path is not None]
    if not resolved:
        raise FileNotFoundError(f"No final plot PNGs found for prefix={prefix!r} in {run_dir}")

    cols = min(2, len(resolved))
    rows_n = ceil(len(resolved) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(10.5 * cols, 4.2 * rows_n), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (row, path) in zip(axes.ravel(), resolved):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        title = (
            f"{row.get('sentence_style', '')} | "
            f"frags={row.get('max_n_fragments', '0')} "
            f"cracked={row.get('max_n_cracked', '0')} "
            f"hoop={_num(row, 'final_crack_hoop_fraction'):.3f}\n"
            f"{_short(row.get('prompt', ''), 72)}"
        )
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fmt(row: dict, key: str) -> str:
    if key not in row:
        return ""
    value = row.get(key, "")
    if key in {
        "final_crack_hoop_fraction",
        "final_crack_nonradial_fraction",
        "final_visited_angular_entropy",
        "max_c_max",
        "elapsed_sec",
    }:
        return f"{_num(row, key):.3f}"
    return str(value)


def write_markdown(rows: list[dict], run_dir: Path, prefix: str, sheet_path: Path, out_path: Path) -> None:
    lines = [
        "# Validation Contact Sheet",
        "",
        f"- Run: `{run_dir}`",
        f"- Prefix: `{prefix}`",
        f"- Contact sheet: `{sheet_path.name}`",
        "",
        f"![contact sheet]({sheet_path.name})",
        "",
        "| prompt | " + " | ".join(METRIC_COLUMNS) + " |",
        "| --- | " + " | ".join("---:" if col not in {"sentence_style", "family", "top1"} else "---" for col in METRIC_COLUMNS) + " |",
    ]
    for row in rows:
        cells = [_short(row.get("prompt", ""), 46)]
        cells.extend(_fmt(row, col) for col in METRIC_COLUMNS)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create contact sheet and markdown summary for a validation output directory.")
    parser.add_argument("--run-dir", required=True, help="Validation output directory.")
    parser.add_argument("--prefix", default="gravity_material_validation", help="CSV/report prefix.")
    parser.add_argument("--out-name", default="validation_contact_sheet", help="Output basename without extension.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    prefix = str(args.prefix)
    csv_path = run_dir / f"{prefix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows = _read_rows(csv_path)
    out_png = run_dir / f"{args.out_name}.png"
    out_md = run_dir / f"{args.out_name}.md"
    write_contact_sheet(rows, run_dir, prefix, out_png)
    write_markdown(rows, run_dir, prefix, out_png, out_md)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
