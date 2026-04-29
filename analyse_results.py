#!/usr/bin/env python3
"""
Analyse network degradation experiment results and generate thesis plots.

Reads summary.csv from network_test.py and produces:
  1. FPS received vs network profile
  2. End-to-end latency vs network profile
  3. Detection delivery rate vs network profile
  4. Combined perception quality score

Usage:
    python3 ~/Desktop/analyse_results.py
    python3 ~/Desktop/analyse_results.py --input network_results/summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication defaults
DPI = 300
FONT_FAMILY = "serif"
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
})


def load_summary(csv_path):
    """Load summary CSV into a list of dicts with numeric conversion."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def has_stddev(rows):
    """Check if the summary contains stddev columns (from repeated trials)."""
    return "meta_fps_std" in rows[0] if rows else False


def compute_quality_score(row):
    """Compute a normalised perception quality score [0-100].

    Weighted combination of:
      - FPS component (40%): normalised to 25 FPS baseline
      - Latency component (30%): penalty for latency above 100ms
      - Delivery rate component (30%): direct percentage
    """
    # FPS: 0-25 FPS maps to 0-100, capped at 100
    fps = row.get("meta_fps", 0)
    fps_score = min(fps / 25.0 * 100, 100)

    # Latency: 0ms = 100, 100ms = 80, 500ms = 40, 1000ms = 10, >1500ms = 0
    lat = row.get("latency_mean_ms", 0)
    if lat <= 0:
        lat_score = 100
    elif lat <= 100:
        lat_score = 100 - (lat / 100) * 20
    elif lat <= 500:
        lat_score = 80 - (lat - 100) / 400 * 40
    elif lat <= 1000:
        lat_score = 40 - (lat - 500) / 500 * 30
    else:
        lat_score = max(10 - (lat - 1000) / 500 * 10, 0)

    # Delivery rate: direct percentage
    dr = row.get("delivery_rate", 0)
    dr_score = dr * 100

    # Weighted combination
    score = 0.4 * fps_score + 0.3 * lat_score + 0.3 * dr_score
    return round(score, 1)


def plot_fps_vs_profile(rows, out_dir, with_err):
    """Bar chart: received FPS for each network profile."""
    labels = [r["label"] for r in rows]
    fps_vals = [r.get("meta_fps", 0) for r in rows]
    fps_err = [r.get("meta_fps_std", 0) for r in rows] if with_err else None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, fps_vals, color="#2196F3", edgecolor="black",
                  linewidth=0.5, yerr=fps_err, capsize=4,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})

    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Frames Per Second (FPS)")
    ax.set_title("Received FPS vs Network Condition", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(y=15, color="red", linestyle="--", alpha=0.7,
               label="Min. requirement (15 FPS)")
    ax.legend(loc="lower left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "fps_vs_profile.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fps_vs_profile.png")


def plot_latency_vs_profile(rows, out_dir, with_err):
    """Bar chart: end-to-end latency for each network profile."""
    labels = [r["label"] for r in rows]
    lat_mean = [r.get("latency_mean_ms", 0) for r in rows]
    lat_p95 = [r.get("latency_p95_ms", r.get("latency_max_ms", 0)) for r in rows]
    lat_mean_err = [r.get("latency_mean_ms_std", 0) for r in rows] if with_err else None
    lat_p95_err = [r.get("latency_p95_ms_std", 0) for r in rows] if with_err else None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, lat_mean, width, label="Mean Latency",
                   color="#4CAF50", edgecolor="black", linewidth=0.5,
                   yerr=lat_mean_err, capsize=3,
                   error_kw={"elinewidth": 1.0, "capthick": 1.0})
    bars2 = ax.bar(x + width / 2, lat_p95, width, label="P95 Latency",
                   color="#FF9800", edgecolor="black", linewidth=0.5,
                   yerr=lat_p95_err, capsize=3,
                   error_kw={"elinewidth": 1.0, "capthick": 1.0})

    for bar, val in zip(bars1, lat_mean):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, lat_p95):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title("Perception Latency vs Network Condition", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(y=250, color="orange", linestyle="--", alpha=0.7,
               label="5G target (250 ms)")
    ax.axhline(y=1000, color="red", linestyle="--", alpha=0.7,
               label="NTN target (1000 ms)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_profile.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: latency_vs_profile.png")


def plot_delivery_vs_profile(rows, out_dir, with_err):
    """Line chart: detection delivery rate vs network profile."""
    labels = [r["label"] for r in rows]
    delivery = [r.get("delivery_rate", 0) * 100 for r in rows]
    delivery_err = [r.get("delivery_rate_std", 0) * 100 for r in rows] if with_err else None
    loss_pct = [r.get("loss_pct", 0) for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))

    color1 = "#9C27B0"
    if with_err and delivery_err:
        ax1.errorbar(x, delivery, yerr=delivery_err, fmt="o-", color=color1,
                     linewidth=2, markersize=8, capsize=4,
                     label="Delivery Rate")
    else:
        ax1.plot(x, delivery, "o-", color=color1, linewidth=2, markersize=8,
                 label="Delivery Rate")
    ax1.set_xlabel("Network Profile")
    ax1.set_ylabel("Frame Delivery Rate (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    color2 = "#F44336"
    ax2.bar(x, loss_pct, alpha=0.3, color=color2, label="Configured Loss (%)")
    ax2.set_ylabel("Packet Loss (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, max(loss_pct) * 2 + 1 if loss_pct else 10)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title("Detection Delivery Rate vs Network Condition",
                  fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_dir / "delivery_vs_profile.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: delivery_vs_profile.png")


def plot_quality_score(rows, out_dir, with_err):
    """Combined perception quality score plot."""
    labels = [r["label"] for r in rows]
    scores = [r.get("quality_score", 0) for r in rows]
    score_err = [r.get("quality_score_std", 0) for r in rows] if with_err else None

    colors = []
    for s in scores:
        if s >= 70:
            colors.append("#4CAF50")
        elif s >= 40:
            colors.append("#FF9800")
        else:
            colors.append("#F44336")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=score_err, capsize=4,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Perception Quality Score (0\u2013100)")
    ax.set_title("Combined Perception Quality vs Network Condition",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 110)

    ax.axhline(y=70, color="green", linestyle="--", alpha=0.5,
               label="Good (> 70)")
    ax.axhline(y=40, color="orange", linestyle="--", alpha=0.5,
               label="Degraded (> 40)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "quality_score.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: quality_score.png")


def print_thesis_table(rows, with_err):
    """Print a formatted table suitable for pasting into a thesis."""
    print(f"\n{'=' * 100}")
    print("  THESIS TABLE — Perception Quality Under Simulated Network Conditions")
    print(f"{'=' * 100}")

    if with_err:
        hdr = (f"  {'Network Condition':<32s} {'Delay':>7s} {'BW':>7s} "
               f"{'Loss':>6s} {'FPS':>12s} {'Latency (ms)':>16s} "
               f"{'Delivery':>12s} {'Score':>8s}")
        sep = (f"  {'─' * 32} {'─' * 7} {'─' * 7} {'─' * 6} "
               f"{'─' * 12} {'─' * 16} {'─' * 12} {'─' * 8}")
        print(hdr)
        print(sep)
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else "∞"
            fps_str = f"{r.get('meta_fps', 0):.1f} ± {r.get('meta_fps_std', 0):.1f}"
            lat_str = f"{r.get('latency_mean_ms', 0):.0f} ± {r.get('latency_mean_ms_std', 0):.0f}"
            dr_val = r.get('delivery_rate', 0)
            dr_std = r.get('delivery_rate_std', 0)
            dr_str = f"{dr_val * 100:.1f} ± {dr_std * 100:.1f}%"
            score_str = f"{r.get('quality_score', 0):.0f}"
            print(f"  {str(r.get('label', '')):<32s} "
                  f"{r.get('delay_ms', 0):>5.0f}ms "
                  f"{bw_str:>6s}M "
                  f"{r.get('loss_pct', 0):>5.1f}% "
                  f"{fps_str:>12s} "
                  f"{lat_str:>16s} "
                  f"{dr_str:>12s} "
                  f"{score_str:>8s}")
    else:
        hdr = (f"  {'Network Condition':<32s} {'Delay':>7s} {'BW':>7s} "
               f"{'Loss':>6s} {'FPS':>6s} {'Lat (ms)':>9s} "
               f"{'P95 (ms)':>9s} {'Delivery':>9s} {'Score':>6s}")
        sep = (f"  {'─' * 32} {'─' * 7} {'─' * 7} {'─' * 6} "
               f"{'─' * 6} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 6}")
        print(hdr)
        print(sep)
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else "∞"
            print(f"  {str(r.get('label', '')):<32s} "
                  f"{r.get('delay_ms', 0):>5.0f}ms "
                  f"{bw_str:>6s}M "
                  f"{r.get('loss_pct', 0):>5.1f}% "
                  f"{r.get('meta_fps', 0):>6.1f} "
                  f"{r.get('latency_mean_ms', 0):>9.1f} "
                  f"{r.get('latency_p95_ms', 0):>9.1f} "
                  f"{r.get('delivery_rate', 0) * 100:>8.1f}% "
                  f"{r.get('quality_score', 0):>5.0f}")

    print(f"{'=' * 100}")

    # Also print a LaTeX-ready version
    print(f"\n  LaTeX table (copy-paste into \\begin{{tabular}}):")
    print(f"  {'─' * 80}")
    if with_err:
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else r"$\infty$"
            print(f"  {r.get('label', '')} & "
                  f"{r.get('delay_ms', 0):.0f} & "
                  f"{bw_str} & "
                  f"{r.get('loss_pct', 0):.1f} & "
                  f"${r.get('meta_fps', 0):.1f} \\pm {r.get('meta_fps_std', 0):.1f}$ & "
                  f"${r.get('latency_mean_ms', 0):.0f} \\pm {r.get('latency_mean_ms_std', 0):.0f}$ & "
                  f"${r.get('delivery_rate', 0) * 100:.1f} \\pm {r.get('delivery_rate_std', 0) * 100:.1f}$ & "
                  f"{r.get('quality_score', 0):.0f} \\\\")
    else:
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else r"$\infty$"
            print(f"  {r.get('label', '')} & "
                  f"{r.get('delay_ms', 0):.0f} & "
                  f"{bw_str} & "
                  f"{r.get('loss_pct', 0):.1f} & "
                  f"{r.get('meta_fps', 0):.1f} & "
                  f"{r.get('latency_mean_ms', 0):.0f} & "
                  f"{r.get('latency_p95_ms', 0):.0f} & "
                  f"{r.get('delivery_rate', 0) * 100:.1f} & "
                  f"{r.get('quality_score', 0):.0f} \\\\")
    print(f"  {'─' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse network degradation experiment results")
    parser.add_argument("--input", default="network_results/summary.csv",
                        help="Path to summary CSV")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for plots (default: same as input)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"[ERROR] Summary CSV not found: {csv_path}\n"
                 f"Run network_test.py first.")

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    rows = load_summary(csv_path)
    print(f"Profiles: {len(rows)}")

    with_err = has_stddev(rows)
    if with_err:
        print("Stddev columns detected — error bars enabled\n")
    else:
        print("Single-trial data — no error bars\n")

    # Compute quality scores
    for r in rows:
        r["quality_score"] = compute_quality_score(r)

    # Generate plots
    print("Generating plots:")
    plot_fps_vs_profile(rows, out_dir, with_err)
    plot_latency_vs_profile(rows, out_dir, with_err)
    plot_delivery_vs_profile(rows, out_dir, with_err)
    plot_quality_score(rows, out_dir, with_err)

    # Update summary CSV with quality scores
    enriched_csv = out_dir / "summary_with_scores.csv"
    with open(enriched_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nEnriched CSV: {enriched_csv}")

    # Print thesis table
    print_thesis_table(rows, with_err)


if __name__ == "__main__":
    main()
