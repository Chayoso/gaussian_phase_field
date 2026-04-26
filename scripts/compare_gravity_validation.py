"""Compare two gravity validation CSV runs.

The intended use is comparing a smaller probe run against a previous reference
run. Absolute particle counts can differ when particle counts differ, so the
report focuses on material/style identity plus relative metric ratios.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IDENTITY_KEYS = ("family", "sentence_style", "top1")
METRIC_KEYS = (
    "max_n_fragments",
    "final_n_fragments",
    "max_n_cracked",
    "final_n_cracked",
    "max_c_max",
    "max_catastrophic_release_nodes",
    "max_physical_fragment_drop",
    "max_detached_distance",
)


def _read_rows(path: Path) -> dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {row["prompt"]: row for row in rows if row.get("prompt")}


def _num(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ratio(candidate: float, baseline: float) -> float | None:
    if abs(baseline) < 1e-12:
        return None
    return candidate / baseline


def compare(baseline_path: Path, candidate_path: Path) -> list[dict]:
    baseline = _read_rows(baseline_path)
    candidate = _read_rows(candidate_path)
    rows: list[dict] = []
    for prompt, base_row in baseline.items():
        cand_row = candidate.get(prompt)
        if cand_row is None:
            rows.append({"prompt": prompt, "status": "missing_candidate"})
            continue

        out = {
            "prompt": prompt,
            "status": "ok",
            "identity_match": all(base_row.get(key, "") == cand_row.get(key, "") for key in IDENTITY_KEYS),
        }
        for key in IDENTITY_KEYS:
            out[f"baseline_{key}"] = base_row.get(key, "")
            out[f"candidate_{key}"] = cand_row.get(key, "")
            out[f"{key}_match"] = base_row.get(key, "") == cand_row.get(key, "")

        for key in METRIC_KEYS:
            base_val = _num(base_row, key)
            cand_val = _num(cand_row, key)
            out[f"baseline_{key}"] = base_val
            out[f"candidate_{key}"] = cand_val
            out[f"{key}_ratio"] = _ratio(cand_val, base_val)
        rows.append(out)

    for prompt in candidate:
        if prompt not in baseline:
            rows.append({"prompt": prompt, "status": "extra_candidate"})
    return rows


def _fmt_ratio(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def write_report(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Gravity Validation Comparison",
        "",
        "Absolute counts are not expected to match across particle counts. Use this as a scale-aware sanity check for material identity and qualitative ordering.",
        "",
        "| prompt | identity | family | style | top1 | frag ratio | cracked ratio | cmax ratio | cat nodes ratio | drop ratio |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('prompt', '')[:52]} | {row.get('status')} |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| {prompt} | {identity} | {family} | {style} | {top1} | {frag} | {cracked} | {cmax} | {cat} | {drop} |".format(
                prompt=row["prompt"][:52],
                identity="yes" if row.get("identity_match") else "no",
                family=row.get("candidate_family", ""),
                style=row.get("candidate_sentence_style", ""),
                top1=row.get("candidate_top1", ""),
                frag=_fmt_ratio(row.get("max_n_fragments_ratio")),
                cracked=_fmt_ratio(row.get("max_n_cracked_ratio")),
                cmax=_fmt_ratio(row.get("max_c_max_ratio")),
                cat=_fmt_ratio(row.get("max_catastrophic_release_nodes_ratio")),
                drop=_fmt_ratio(row.get("max_physical_fragment_drop_ratio")),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gravity validation CSV files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True, help="Output directory for comparison CSV/MD/JSON.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = compare(Path(args.baseline), Path(args.candidate))
    (out_dir / "gravity_comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(rows, out_dir / "gravity_comparison.csv")
    write_report(rows, out_dir / "gravity_comparison.md")
    print(f"Wrote comparison: {out_dir}")


if __name__ == "__main__":
    main()
