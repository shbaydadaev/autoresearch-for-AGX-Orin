"""
Lightweight terminal-friendly analysis for autoresearch experiment logs.

Reads results.tsv, prints summary stats, and regenerates a progress plot.

Usage:
    python3 analyze_results.py
    python3 analyze_results.py --results results.tsv --plot autoresearch_progress.png
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["val_bpb"] = float(row["val_bpb"])
            row["memory_gb"] = float(row["memory_gb"])
            row["status"] = row["status"].strip().upper()
            rows.append(row)
    return rows


def print_summary(rows):
    print(f"Total experiments: {len(rows)}")
    counts = Counter(row["status"] for row in rows)
    print("Outcome counts:")
    for key in ["KEEP", "DISCARD", "CRASH"]:
        print(f"  {key:7s} {counts.get(key, 0)}")

    decided = counts.get("KEEP", 0) + counts.get("DISCARD", 0)
    if decided:
        keep_rate = counts.get("KEEP", 0) / decided
        print(f"Keep rate: {counts.get('KEEP', 0)}/{decided} = {keep_rate:.1%}")

    baseline = rows[0]
    print()
    print("Baseline:")
    print(f"  val_bpb={baseline['val_bpb']:.6f} mem={baseline['memory_gb']:.1f}GB desc={baseline['description']}")

    kept = [row for row in rows if row["status"] == "KEEP"]
    if kept:
        best = min(kept, key=lambda row: row["val_bpb"])
        print()
        print("Best kept result:")
        print(f"  val_bpb={best['val_bpb']:.6f} mem={best['memory_gb']:.1f}GB desc={best['description']}")
        improvement = baseline["val_bpb"] - best["val_bpb"]
        print(f"  improvement={improvement:+.6f}")

    if kept:
        print()
        print("Kept experiments:")
        for idx, row in enumerate(rows):
            if row["status"] == "KEEP":
                print(f"  #{idx:03d}  {row['val_bpb']:.6f}  {row['memory_gb']:.1f}GB  {row['description']}")


def plot_progress(rows, out_path):
    if plt is None:
        return plot_progress_svg(rows, out_path)

    x = list(range(len(rows)))
    y = [row["val_bpb"] for row in rows]
    labels = [row["description"] for row in rows]
    statuses = [row["status"] for row in rows]

    color_map = {
        "KEEP": "#2ecc71",
        "DISCARD": "#bdbdbd",
        "CRASH": "#e74c3c",
    }
    colors = [color_map.get(status, "#1f77b4") for status in statuses]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color="#1f77b4", linewidth=1.8, alpha=0.65, zorder=1)
    ax.scatter(x, y, c=colors, s=70, edgecolors="black", linewidths=0.5, zorder=3)

    kept_x = []
    kept_y = []
    running_min = []
    current_best = None
    for idx, row in enumerate(rows):
        if row["status"] == "KEEP":
            kept_x.append(idx)
            kept_y.append(row["val_bpb"])
            current_best = row["val_bpb"] if current_best is None else min(current_best, row["val_bpb"])
            running_min.append(current_best)
    if kept_x:
        ax.step(kept_x, running_min, where="post", color="#27ae60", linewidth=2.0, alpha=0.8, zorder=2)

    for idx, row in enumerate(rows):
        label = str(labels[idx]).strip()
        if len(label) > 28:
            label = label[:25] + "..."
        ax.annotate(label, (idx, y[idx]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)

    ax.set_title("Autoresearch Progress: 19 Experiments, 6 Kept", fontsize=16)
    ax.set_xlabel("Experiment Index")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    return Path(out_path)


def plot_progress_svg(rows, out_path):
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")

    width = 1000
    height = 520
    left = 80
    right = 30
    top = 40
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = [row["val_bpb"] for row in rows]
    min_y = min(values)
    max_y = max(values)
    if min_y == max_y:
        min_y -= 0.05
        max_y += 0.05
    pad = max((max_y - min_y) * 0.08, 0.02)
    min_y -= pad
    max_y += pad

    def x_pos(idx):
        if len(rows) == 1:
            return left + plot_w / 2
        return left + idx * (plot_w / (len(rows) - 1))

    def y_pos(value):
        frac = (value - min_y) / (max_y - min_y)
        return top + plot_h - frac * plot_h

    color_map = {
        "KEEP": "#2ecc71",
        "DISCARD": "#bdbdbd",
        "CRASH": "#e74c3c",
    }

    points = [(x_pos(i), y_pos(row["val_bpb"])) for i, row in enumerate(rows)]
    polyline_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    running_best = []
    current_best = None
    for idx, row in enumerate(rows):
        if row["status"] == "KEEP":
            current_best = row["val_bpb"] if current_best is None else min(current_best, row["val_bpb"])
            running_best.append((x_pos(idx), y_pos(current_best)))
    running_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in running_best)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.0f}" y="24" text-anchor="middle" font-family="sans-serif" font-size="22">Autoresearch Progress: 19 Experiments, 6 Kept</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.2"/>',
    ]

    for tick_idx in range(5):
        frac = tick_idx / 4
        value = min_y + frac * (max_y - min_y)
        y = y_pos(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3f}</text>')

    for idx in range(len(rows)):
        x = x_pos(idx)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f2f2f2" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{idx}</text>')

    lines.append(f'<polyline points="{polyline_points}" fill="none" stroke="#1f77b4" stroke-width="2" opacity="0.75"/>')
    if running_points:
        lines.append(f'<polyline points="{running_points}" fill="none" stroke="#27ae60" stroke-width="2.5" opacity="0.85"/>')

    for idx, row in enumerate(rows):
        x, y = points[idx]
        color = color_map.get(row["status"], "#1f77b4")
        label = str(row["description"]).strip()
        if len(label) > 28:
            label = label[:25] + "..."
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="black" stroke-width="0.8"/>')
        lines.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>')

    lines.extend([
        f'<text x="{width/2:.0f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">Experiment Index</text>',
        f'<text x="22" y="{height/2:.0f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 22,{height/2:.0f})">Validation BPB (lower is better)</text>',
        '</svg>',
    ])

    out_path.write_text("\n".join(lines))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Analyze autoresearch results.tsv")
    parser.add_argument("--results", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--plot", default="autoresearch_progress.png", help="Output plot path")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    rows = load_results(results_path)
    if not rows:
        raise SystemExit("Results file is empty.")

    print_summary(rows)
    saved_path = plot_progress(rows, args.plot)
    print()
    print(f"Saved plot: {Path(saved_path).resolve()}")


if __name__ == "__main__":
    main()
