#!/usr/bin/env python3
"""
Analyse network degradation experiment results and generate thesis plots.

Produces 6 publication-quality plots at 300 DPI:
  1. fps_vs_profile.png       — grouped bars: video FPS vs metadata FPS
  2. latency_vs_profile.png   — mean and P95 with error bars
  3. delivery_vs_profile.png  — video / meta / lidar delivery
  4. throughput_vs_profile.png — video and LIDAR throughput (bytes/sec)
  5. quality_score.png        — combined perception quality
  6. server_vs_client.png     — frames sent vs received per profile
                                 (when server_log.csv is available)

Usage:
    python3 ~/Desktop/analyse_results.py
    python3 ~/Desktop/analyse_results.py --input network_results/full_experiment_v2/summary.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 300
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
})


def load_summary(csv_path):
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
    return "meta_fps_std" in rows[0] if rows else False


def compute_quality_score(row):
    """Quality score [0-100] weighted: 40% video FPS, 30% latency, 30% delivery."""
    fps = row.get("video_fps", row.get("meta_fps", 0))
    fps_score = min(fps / 25.0 * 100, 100)

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

    dr = row.get("delivery_rate", 0)
    dr_score = dr * 100

    return round(0.4 * fps_score + 0.3 * lat_score + 0.3 * dr_score, 1)


def plot_fps_vs_profile(rows, out_dir, with_err):
    """Grouped bars: video FPS vs metadata FPS per profile."""
    labels = [r["label"] for r in rows]
    video_fps = [r.get("video_fps", 0) for r in rows]
    meta_fps = [r.get("meta_fps", 0) for r in rows]
    video_err = [r.get("video_fps_std", 0) for r in rows] if with_err else None
    meta_err = [r.get("meta_fps_std", 0) for r in rows] if with_err else None

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    width = 0.38

    b1 = ax.bar(x - width / 2, video_fps, width, label="Video (MJPEG)",
                color="#2196F3", edgecolor="black", linewidth=0.5,
                yerr=video_err, capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0})
    b2 = ax.bar(x + width / 2, meta_fps, width, label="Metadata (JSON)",
                color="#4CAF50", edgecolor="black", linewidth=0.5,
                yerr=meta_err, capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0})

    for bar, val in zip(b1, video_fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, meta_fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Frames per Second")
    ax.set_title("Video vs Metadata Throughput by Network Condition",
                 fontweight="bold")
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
    labels = [r["label"] for r in rows]
    lat_mean = [r.get("latency_mean_ms", 0) for r in rows]
    lat_p95 = [r.get("latency_p95_ms", r.get("latency_max_ms", 0)) for r in rows]
    lat_mean_err = [r.get("latency_mean_ms_std", 0) for r in rows] if with_err else None
    lat_p95_err = [r.get("latency_p95_ms_std", 0) for r in rows] if with_err else None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35

    b1 = ax.bar(x - width / 2, lat_mean, width, label="Mean Latency",
                color="#4CAF50", edgecolor="black", linewidth=0.5,
                yerr=lat_mean_err, capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0})
    b2 = ax.bar(x + width / 2, lat_p95, width, label="P95 Latency",
                color="#FF9800", edgecolor="black", linewidth=0.5,
                yerr=lat_p95_err, capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0})

    for bar, val in zip(b1, lat_mean):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, lat_p95):
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
    """Three streams: video, metadata, lidar — receive rate normalised."""
    labels = [r["label"] for r in rows]

    # Compute per-stream relative rate (vs baseline = profile 0 of the same stream)
    def _rel(key):
        baseline = rows[0].get(key, 0)
        if baseline <= 0:
            return [0] * len(rows)
        return [r.get(key, 0) / baseline * 100 for r in rows]

    video_rel = _rel("video_fps")
    meta_rel = _rel("meta_fps")
    lidar_rel = _rel("lidar_fps")
    has_lidar = any(r.get("lidar_fps", 0) > 0 for r in rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))

    ax.plot(x, video_rel, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="Video (MJPEG)")
    ax.plot(x, meta_rel, "s-", color="#4CAF50", linewidth=2, markersize=8,
            label="Metadata (JSON)")
    if has_lidar:
        ax.plot(x, lidar_rel, "^-", color="#9C27B0", linewidth=2, markersize=8,
                label="LIDAR scans")

    # Configured loss as faint background bars
    loss_pct = [r.get("loss_pct", 0) for r in rows]
    ax2 = ax.twinx()
    ax2.bar(x, loss_pct, alpha=0.15, color="#F44336",
            label="Configured Loss (%)")
    ax2.set_ylabel("Configured Packet Loss (%)", color="#F44336")
    ax2.tick_params(axis="y", labelcolor="#F44336")
    ax2.set_ylim(0, max(loss_pct) * 2 + 1 if any(loss_pct) else 10)

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Throughput Relative to Baseline (%)")
    ax.set_title("Per-Stream Degradation vs Network Condition",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="black", linestyle=":", alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_dir / "delivery_vs_profile.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: delivery_vs_profile.png")


def plot_throughput_vs_profile(rows, out_dir):
    """Effective throughput (KiB/s) for video and LIDAR."""
    labels = [r["label"] for r in rows]
    video_kib = [r.get("video_bytes_per_sec", 0) / 1024 for r in rows]
    lidar_kib = [r.get("lidar_bytes_per_sec", 0) / 1024 for r in rows]
    has_lidar = any(v > 0 for v in lidar_kib)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))

    ax.plot(x, video_kib, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="Video (MJPEG)")
    if has_lidar:
        ax.plot(x, lidar_kib, "^-", color="#9C27B0", linewidth=2, markersize=8,
                label="LIDAR scans")

    for xi, val in zip(x, video_kib):
        ax.text(xi, val + max(video_kib) * 0.02, f"{val:.0f}",
                ha="center", va="bottom", fontsize=8, color="#1565C0")
    if has_lidar:
        for xi, val in zip(x, lidar_kib):
            ax.text(xi, val + max(video_kib) * 0.02, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8, color="#6A1B9A")

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Effective Throughput (KiB/s)")
    ax.set_title("Effective Stream Throughput vs Network Condition",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_profile.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: throughput_vs_profile.png")


def plot_quality_score(rows, out_dir, with_err):
    labels = [r["label"] for r in rows]
    scores = [r.get("quality_score", 0) for r in rows]
    score_err = [r.get("quality_score_std", 0) for r in rows] if with_err else None

    colors = ["#4CAF50" if s >= 70 else "#FF9800" if s >= 40 else "#F44336"
              for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=score_err, capsize=4,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Perception Quality Score (0–100)")
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


def server_vs_client(rows, exp_dir, server_log_path, out_dir):
    """Compare server-sent vs client-received frames for each profile.

    Reads run_start_ts/run_end_ts from each profile.json (per-repeat windows),
    counts server_log entries within those windows, and compares to received
    metadata count.
    """
    if not server_log_path.exists():
        print(f"  [SKIP] Server log not found: {server_log_path}")
        return

    server_events = []
    with open(server_log_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                server_events.append(float(r["timestamp"]))
            except (ValueError, KeyError):
                pass
    if not server_events:
        print(f"  [SKIP] Server log empty: {server_log_path}")
        return
    server_events.sort()
    print(f"  Server log: {len(server_events)} frames spanning "
          f"{server_events[-1] - server_events[0]:.1f}s")

    sent = []
    received = []
    labels = []

    for r in rows:
        profile_key = r.get("profile", "")
        labels.append(r["label"])
        prof_dir = exp_dir / str(profile_key)
        prof_json = prof_dir / "profile.json"
        if not prof_json.exists():
            sent.append(0)
            received.append(0)
            continue

        with open(prof_json) as f:
            prof_data = json.load(f)

        repeats = prof_data.get("repeats", [])
        total_sent = 0
        total_received = 0
        for rep in repeats:
            t0 = rep.get("run_start_ts")
            t1 = rep.get("run_end_ts")
            if not t0 or not t1:
                continue
            n_sent = sum(1 for ts in server_events if t0 <= ts <= t1)
            n_recv = rep.get("meta_messages", 0)
            total_sent += n_sent
            total_received += n_recv
        sent.append(total_sent)
        received.append(total_received)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width / 2, sent, width, label="Server-sent",
           color="#1976D2", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, received, width, label="Client-received",
           color="#43A047", edgecolor="black", linewidth=0.5)

    for xi, s_val, r_val in zip(x, sent, received):
        if s_val > 0:
            loss = (s_val - r_val) / s_val * 100 if s_val else 0
            ax.text(xi, max(s_val, r_val) * 1.02, f"−{loss:.1f}%",
                    ha="center", va="bottom", fontsize=8,
                    color="red" if loss > 5 else "black")

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Frame Count (across all repeats)")
    ax.set_title("Server-Sent vs Client-Received Detection Frames",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "server_vs_client.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: server_vs_client.png")

    print("\n  Server-sent vs client-received frames per profile:")
    print(f"  {'Profile':<32s} {'Sent':>8s} {'Received':>10s} {'Lost':>10s}")
    print(f"  {'─' * 32} {'─' * 8} {'─' * 10} {'─' * 10}")
    for lab, s_val, r_val in zip(labels, sent, received):
        loss_pct = (s_val - r_val) / s_val * 100 if s_val else 0
        print(f"  {lab:<32s} {s_val:>8d} {r_val:>10d} "
              f"{loss_pct:>9.2f}%")


def print_thesis_table(rows, with_err):
    print(f"\n{'=' * 115}")
    print("  THESIS TABLE — Perception Quality Under Simulated Network Conditions")
    print(f"{'=' * 115}")

    if with_err:
        hdr = (f"  {'Network Condition':<32s} {'Delay':>7s} {'BW':>7s} "
               f"{'Loss':>5s} {'Video FPS':>14s} {'Meta FPS':>12s} "
               f"{'Latency (ms)':>16s} {'Score':>6s}")
        print(hdr)
        print(f"  {'─' * 32} {'─' * 7} {'─' * 7} {'─' * 5} "
              f"{'─' * 14} {'─' * 12} {'─' * 16} {'─' * 6}")
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else "∞"
            v_str = f"{r.get('video_fps', 0):.1f}±{r.get('video_fps_std', 0):.1f}"
            m_str = f"{r.get('meta_fps', 0):.1f}±{r.get('meta_fps_std', 0):.1f}"
            lat_str = f"{r.get('latency_mean_ms', 0):.0f}±{r.get('latency_mean_ms_std', 0):.0f}"
            score_str = f"{r.get('quality_score', 0):.0f}"
            print(f"  {str(r.get('label', '')):<32s} "
                  f"{r.get('delay_ms', 0):>5.0f}ms "
                  f"{bw_str:>6s}M "
                  f"{r.get('loss_pct', 0):>4.1f}% "
                  f"{v_str:>14s} "
                  f"{m_str:>12s} "
                  f"{lat_str:>16s} "
                  f"{score_str:>6s}")
    else:
        hdr = (f"  {'Network Condition':<32s} {'Delay':>7s} {'BW':>7s} "
               f"{'Loss':>5s} {'Video FPS':>10s} {'Meta FPS':>9s} "
               f"{'Lat (ms)':>9s} {'P95 (ms)':>9s} {'Score':>6s}")
        print(hdr)
        print(f"  {'─' * 32} {'─' * 7} {'─' * 7} {'─' * 5} "
              f"{'─' * 10} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 6}")
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else "∞"
            print(f"  {str(r.get('label', '')):<32s} "
                  f"{r.get('delay_ms', 0):>5.0f}ms "
                  f"{bw_str:>6s}M "
                  f"{r.get('loss_pct', 0):>4.1f}% "
                  f"{r.get('video_fps', 0):>10.1f} "
                  f"{r.get('meta_fps', 0):>9.1f} "
                  f"{r.get('latency_mean_ms', 0):>9.1f} "
                  f"{r.get('latency_p95_ms', 0):>9.1f} "
                  f"{r.get('quality_score', 0):>6.0f}")
    print(f"{'=' * 115}")

    # LaTeX table
    print(f"\n  LaTeX rows (paste into \\begin{{tabular}}):")
    print(f"  {'─' * 80}")
    if with_err:
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else r"$\infty$"
            print(f"  {r.get('label', '')} & "
                  f"{r.get('delay_ms', 0):.0f} & "
                  f"{bw_str} & "
                  f"{r.get('loss_pct', 0):.1f} & "
                  f"${r.get('video_fps', 0):.1f} \\pm {r.get('video_fps_std', 0):.1f}$ & "
                  f"${r.get('meta_fps', 0):.1f} \\pm {r.get('meta_fps_std', 0):.1f}$ & "
                  f"${r.get('latency_mean_ms', 0):.0f} \\pm {r.get('latency_mean_ms_std', 0):.0f}$ & "
                  f"{r.get('quality_score', 0):.0f} \\\\")
    else:
        for r in rows:
            bw = r.get("bandwidth_mbps", 0)
            bw_str = f"{bw:.1f}" if bw > 0 else r"$\infty$"
            print(f"  {r.get('label', '')} & "
                  f"{r.get('delay_ms', 0):.0f} & "
                  f"{bw_str} & "
                  f"{r.get('loss_pct', 0):.1f} & "
                  f"{r.get('video_fps', 0):.1f} & "
                  f"{r.get('meta_fps', 0):.1f} & "
                  f"{r.get('latency_mean_ms', 0):.0f} & "
                  f"{r.get('latency_p95_ms', 0):.0f} & "
                  f"{r.get('quality_score', 0):.0f} \\\\")
    print(f"  {'─' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse network degradation experiment results")
    parser.add_argument("--input", default="network_results/summary.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--server-log", default=None,
                        help="Path to server_log.csv (for server-vs-client comparison)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"[ERROR] Summary CSV not found: {csv_path}")

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    exp_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default server log location: experiment dir, then parent
    if args.server_log:
        server_log_path = Path(args.server_log)
    elif (exp_dir / "server_log.csv").exists():
        server_log_path = exp_dir / "server_log.csv"
    else:
        server_log_path = exp_dir.parent / "server_log.csv"

    print(f"Loading: {csv_path}")
    rows = load_summary(csv_path)
    print(f"Profiles: {len(rows)}")

    with_err = has_stddev(rows)
    if with_err:
        print("Stddev columns detected — error bars enabled\n")
    else:
        print("Single-trial data — no error bars\n")

    for r in rows:
        r["quality_score"] = compute_quality_score(r)

    print("Generating plots:")
    plot_fps_vs_profile(rows, out_dir, with_err)
    plot_latency_vs_profile(rows, out_dir, with_err)
    plot_delivery_vs_profile(rows, out_dir, with_err)
    plot_throughput_vs_profile(rows, out_dir)
    plot_quality_score(rows, out_dir, with_err)
    server_vs_client(rows, exp_dir, server_log_path, out_dir)

    enriched_csv = out_dir / "summary_with_scores.csv"
    with open(enriched_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nEnriched CSV: {enriched_csv}")

    print_thesis_table(rows, with_err)


if __name__ == "__main__":
    main()
