#!/usr/bin/env python3
"""
Benchmark Results Visualization Script

Generates matplotlib charts comparing native vs pytesseract benchmark results
across different dataset versions.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Style configuration for all engine configurations
# Colors chosen for maximum contrast:
# - Baseline versions: cooler colors (green, blue)
# - Optimized versions: warmer colors (orange, purple)
ENGINE_COLORS = {
    "native": "#2ecc71",           # Green (baseline)
    "native-optimized": "#e74c3c", # Red (optimized - high contrast)
    "pytesseract": "#3498db",      # Blue (baseline)
    "pytesseract-optimized": "#9b59b6",  # Purple (optimized - high contrast)
}

ENGINE_LINESTYLES = {
    "native": "-",
    "native-optimized": "-.",      # Dash-dot for optimized
    "pytesseract": "--",
    "pytesseract-optimized": ":",  # Dotted for optimized
}

ENGINE_MARKERS = {
    "native": "o",
    "native-optimized": "^",
    "pytesseract": "s",
    "pytesseract-optimized": "D",
}

# Display names for cleaner labels
ENGINE_DISPLAY_NAMES = {
    "native": "Native CLI",
    "native-optimized": "Native CLI + OMP_THREAD_LIMIT",
    "pytesseract": "Pytesseract",
    "pytesseract-optimized": "Pytesseract + OMP_THREAD_LIMIT",
}


def get_engine_style(engine: str) -> dict:
    """Get consistent styling for an engine configuration."""
    return {
        "color": ENGINE_COLORS.get(engine, "#95a5a6"),
        "linestyle": ENGINE_LINESTYLES.get(engine, "-"),
        "marker": ENGINE_MARKERS.get(engine, "o"),
        "label": ENGINE_DISPLAY_NAMES.get(engine, engine),
    }


def filter_results_by_engines(results: dict, engines: list[str] | None) -> dict:
    """Filter results to only include specified engine configurations."""
    if engines is None:
        return results

    filtered = {}
    for version, version_data in results.items():
        filtered_version = {
            eng: data for eng, data in version_data.items()
            if eng in engines
        }
        if filtered_version:
            filtered[version] = filtered_version

    return filtered


def load_benchmark_results(results_dir: Path) -> dict:
    """
    Load all benchmark results organized by version and engine configuration.

    Detects 'optimized' pattern in filenames to create separate engine configurations:
    - native, native-optimized, pytesseract, pytesseract-optimized

    Returns:
        dict: {version: {engine_config: [results...]}}
    """
    results = defaultdict(lambda: defaultdict(list))

    for version_dir in sorted(results_dir.iterdir()):
        if not version_dir.is_dir():
            continue

        version = version_dir.name

        for engine_dir in version_dir.iterdir():
            if not engine_dir.is_dir():
                continue

            base_engine = engine_dir.name

            for json_file in sorted(engine_dir.glob("*.json")):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        data["_source_file"] = str(json_file)

                        # Detect optimized variant from filename
                        filename = json_file.stem.lower()
                        if "optimized" in filename:
                            engine_config = f"{base_engine}-optimized"
                        else:
                            engine_config = base_engine

                        results[version][engine_config].append(data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load {json_file}: {e}", file=sys.stderr)

    return dict(results)


def plot_resource_timeline(
    results: dict,
    version: str,
    output_dir: Path,
    suffix: str = ""
) -> Optional[Figure]:
    """
    Plot CPU and memory usage over time for both engines in a version.
    """
    version_data = results.get(version, {})
    if not version_data:
        print(f"No data for version {version}", file=sys.stderr)
        return None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    engines_in_plot = sorted(version_data.keys())
    fig.suptitle(f"Resource Usage Timeline - Dataset {version}\n({', '.join(engines_in_plot)})", fontsize=14, fontweight="bold")

    for engine, runs in version_data.items():
        if not runs:
            continue

        # Use the most recent run
        run = runs[-1]
        samples = run.get("resource_samples", [])

        if not samples:
            continue

        # Normalize timestamps to start from 0
        base_ts = samples[0]["timestamp"]
        times = [(s["timestamp"] - base_ts) for s in samples]
        cpu = [s["cpu_percent"] for s in samples]
        memory = [s["memory_mb"] for s in samples]

        style = get_engine_style(engine)
        label = f"{engine} (peak CPU: {run['peak_cpu_percent']:.1f}%, peak mem: {run['peak_memory_mb']:.1f} MB)"

        axes[0].plot(times, cpu, color=style["color"], linestyle=style["linestyle"], label=label, linewidth=1.5, alpha=0.8)
        axes[1].plot(times, memory, color=style["color"], linestyle=style["linestyle"], label=engine, linewidth=1.5, alpha=0.8)

    axes[0].set_ylabel("CPU Usage (%)")
    axes[0].set_title("CPU Usage Over Time")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].set_ylabel("Memory (MB)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_title("Memory Usage Over Time")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()

    output_file = output_dir / f"resource_timeline_{version}{suffix}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    return fig


def plot_pages_over_time(
    results: dict,
    version: str,
    output_dir: Path,
    suffix: str = ""
) -> Optional[Figure]:
    """
    Plot cumulative pages processed over time for both engines.
    """
    version_data = results.get(version, {})
    if not version_data:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    engines_in_plot = sorted(version_data.keys())
    fig.suptitle(f"Pages Processed Over Time - Dataset {version}\n({', '.join(engines_in_plot)})", fontsize=14, fontweight="bold")

    for engine, runs in version_data.items():
        if not runs:
            continue

        run = runs[-1]
        ocr_results = run.get("ocr_results", [])

        if not ocr_results:
            continue

        # Collect all page end timestamps and sort them
        page_events = []
        for result in ocr_results:
            page_metrics = result.get("page_metrics", [])
            if not page_metrics:
                continue
            for page in page_metrics:
                page_events.append(page["end_timestamp"])

        if not page_events:
            continue

        page_events.sort()

        # Normalize to start from first page
        base_ts = page_events[0]
        times = [0] + [(ts - base_ts) for ts in page_events]
        pages = list(range(len(page_events) + 1))

        style = get_engine_style(engine)

        total_time = run.get("total_duration_s", times[-1])
        pages_per_sec = len(page_events) / total_time if total_time > 0 else 0
        label = f"{engine} ({len(page_events)} pages, {pages_per_sec:.2f} pages/sec)"

        ax.step(times, pages, where="post", color=style["color"], linewidth=2, label=label, alpha=0.8)
        ax.scatter(times[1:], pages[1:], color=style["color"], marker=style["marker"], s=30, alpha=0.6)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cumulative Pages Processed")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    output_file = output_dir / f"pages_over_time_{version}{suffix}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    return fig


def plot_processing_time_per_page(
    results: dict,
    version: str,
    output_dir: Path,
    suffix: str = ""
) -> Optional[Figure]:
    """
    Bar chart comparing processing time per page for each file.
    """
    version_data = results.get(version, {})
    if not version_data:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    engines_in_plot = sorted(version_data.keys())
    fig.suptitle(f"Average Processing Time per Page - Dataset {version}\n({', '.join(engines_in_plot)})", fontsize=14, fontweight="bold")

    # Collect data
    engine_data = {}
    all_files = set()

    for engine, runs in version_data.items():
        if not runs:
            continue

        run = runs[-1]
        avg_times = run.get("avg_time_per_page", [])

        engine_data[engine] = {}
        for item in avg_times:
            filename = item["filename"]
            all_files.add(filename)
            engine_data[engine][filename] = item["avg_time_per_page_ms"]

    if not all_files:
        return None

    all_files = sorted(all_files)
    x = range(len(all_files))

    engines = sorted(engine_data.keys())
    # Adjust bar width based on number of engines
    width = 0.8 / len(engines) if engines else 0.35

    for i, engine in enumerate(engines):
        offset = (i - len(engines) / 2 + 0.5) * width
        values = [engine_data[engine].get(f, 0) for f in all_files]
        style = get_engine_style(engine)
        bars = ax.bar([xi + offset for xi in x], values, width, label=engine, color=style["color"], alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(
                    f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom",
                    fontsize=8
                )

    ax.set_xlabel("File")
    ax.set_ylabel("Time per Page (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".pdf", "") for f in all_files], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_file = output_dir / f"time_per_page_{version}{suffix}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    return fig


def plot_comparison_across_versions(
    results: dict,
    output_dir: Path,
    suffix: str = ""
) -> Optional[Figure]:
    """
    Compare key metrics across all versions for both engines.
    """
    if len(results) < 1:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    versions = sorted(results.keys())
    engines = set()
    for v in versions:
        engines.update(results[v].keys())
    engines = sorted(engines)

    fig.suptitle(f"Benchmark Comparison Across Dataset Versions\n({', '.join(engines)})", fontsize=14, fontweight="bold")

    # Collect metrics
    metrics = {
        "total_duration_s": {"title": "Total Duration (s)", "ax": axes[0, 0]},
        "peak_memory_mb": {"title": "Peak Memory (MB)", "ax": axes[0, 1]},
        "peak_cpu_percent": {"title": "Peak CPU (%)", "ax": axes[1, 0]},
        "pages_per_sec": {"title": "Pages per Second", "ax": axes[1, 1]},
    }

    for engine in engines:
        style = get_engine_style(engine)
        for metric_key, metric_info in metrics.items():
            values = []
            valid_versions = []

            for version in versions:
                runs = results.get(version, {}).get(engine, [])
                if not runs:
                    continue

                run = runs[-1]

                if metric_key == "pages_per_sec":
                    total_pages = sum(r.get("page_count", 0) for r in run.get("ocr_results", []))
                    duration = run.get("total_duration_s", 1)
                    value = total_pages / duration if duration > 0 else 0
                else:
                    value = run.get(metric_key, 0)

                values.append(value)
                valid_versions.append(version)

            if values:
                metric_info["ax"].plot(
                    valid_versions, values,
                    color=style["color"], marker=style["marker"], linewidth=2,
                    label=engine, markersize=8
                )

    for metric_info in metrics.values():
        ax = metric_info["ax"]
        ax.set_title(metric_info["title"])
        ax.set_xlabel("Dataset Version")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    output_file = output_dir / f"comparison_across_versions{suffix}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    return fig


def plot_page_level_metrics(
    results: dict,
    version: str,
    output_dir: Path,
    suffix: str = ""
) -> Optional[Figure]:
    """
    Plot per-page processing time showing page dimensions impact.
    """
    version_data = results.get(version, {})
    if not version_data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    engines_in_plot = sorted(version_data.keys())
    fig.suptitle(f"Page-Level Metrics - Dataset {version}\n({', '.join(engines_in_plot)})", fontsize=14, fontweight="bold")

    # Collect data for both scatter and boxplot
    bp_data = []
    bp_labels = []

    for engine in sorted(version_data.keys()):
        runs = version_data[engine]
        if not runs:
            continue

        run = runs[-1]
        ocr_results = run.get("ocr_results", [])

        page_sizes = []
        proc_times = []

        for result in ocr_results:
            page_metrics = result.get("page_metrics", [])
            for page in page_metrics:
                # Calculate page area in megapixels
                area_mp = (page["width"] * page["height"]) / 1_000_000
                page_sizes.append(area_mp)
                proc_times.append(page["processing_time_ms"])

        if not page_sizes:
            continue

        style = get_engine_style(engine)

        # Scatter: page size vs processing time
        axes[0].scatter(
            page_sizes, proc_times,
            color=style["color"], alpha=0.6, label=engine, s=60,
            marker=style["marker"]
        )

        # Collect for boxplot
        bp_data.append(proc_times)
        bp_labels.append(engine)

    axes[0].set_xlabel("Page Size (Megapixels)")
    axes[0].set_ylabel("Processing Time (ms)")
    axes[0].set_title("Processing Time vs Page Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    if bp_data:
        bp = axes[1].boxplot(bp_data, tick_labels=bp_labels, patch_artist=True)
        for patch, label in zip(bp["boxes"], bp_labels):
            style = get_engine_style(label)
            patch.set_facecolor(style["color"])
            patch.set_alpha(0.6)

    axes[1].set_ylabel("Processing Time (ms)")
    axes[1].set_title("Processing Time Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_file = output_dir / f"page_metrics_{version}{suffix}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    return fig


def plot_optimization_impact(
    results: dict,
    version: str,
    output_dir: Path,
    suffix: str = ""
) -> Optional[Figure]:
    """
    Create a dashboard showing optimization impact on resource usage.
    Helps decide whether to implement optimizations in production.
    """
    version_data = results.get(version, {})
    if not version_data:
        return None

    # Define comparison pairs (baseline -> optimized)
    comparison_pairs = [
        ("native", "native-optimized"),
        ("pytesseract", "pytesseract-optimized"),
    ]

    # Filter to pairs that exist in data
    valid_pairs = [
        (base, opt) for base, opt in comparison_pairs
        if base in version_data and opt in version_data
    ]

    if not valid_pairs:
        return None

    # Metrics to compare (key, label, unit, lower_is_better)
    metrics = [
        ("peak_memory_mb", "Peak Memory", "MB", True),
        ("avg_memory_mb", "Avg Memory", "MB", True),
        ("peak_cpu_percent", "Peak CPU", "%", True),
        ("total_duration_s", "Total Duration", "s", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Optimization Impact Analysis - Dataset {version}\n"
        f"(OMP_THREAD_LIMIT=1 Effect on Resource Usage)",
        fontsize=14, fontweight="bold"
    )

    axes_flat = axes.flatten()

    for idx, (metric_key, metric_label, unit, lower_is_better) in enumerate(metrics):
        ax = axes_flat[idx]

        x_positions = []
        bar_width = 0.35
        x = 0

        for base_engine, opt_engine in valid_pairs:
            base_run = version_data[base_engine][-1]
            opt_run = version_data[opt_engine][-1]

            base_value = base_run.get(metric_key, 0)
            opt_value = opt_run.get(metric_key, 0)

            # Calculate improvement
            if base_value > 0:
                change_pct = ((base_value - opt_value) / base_value) * 100
            else:
                change_pct = 0

            base_style = get_engine_style(base_engine)
            opt_style = get_engine_style(opt_engine)

            # Draw bars
            bar1 = ax.bar(x - bar_width/2, base_value, bar_width,
                         label=base_style["label"] if idx == 0 else "",
                         color=base_style["color"], alpha=0.8, edgecolor="black")
            bar2 = ax.bar(x + bar_width/2, opt_value, bar_width,
                         label=opt_style["label"] if idx == 0 else "",
                         color=opt_style["color"], alpha=0.8, edgecolor="black")

            # Add value labels on bars
            ax.bar_label(bar1, fmt=f'%.1f', fontsize=9, fontweight="bold")
            ax.bar_label(bar2, fmt=f'%.1f', fontsize=9, fontweight="bold")

            # Add change indicator
            is_improvement = (change_pct > 0) if lower_is_better else (change_pct < 0)
            change_color = "#27ae60" if is_improvement else "#e74c3c"
            change_symbol = "↓" if change_pct > 0 else "↑"

            max_val = max(base_value, opt_value)
            ax.annotate(
                f"{change_symbol} {abs(change_pct):.1f}%",
                xy=(x, max_val * 1.1),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=change_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=change_color, alpha=0.9)
            )

            x_positions.append(x)
            x += 1.5

        ax.set_ylabel(f"{metric_label} ({unit})")
        ax.set_title(metric_label, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{base.replace('-optimized', '')}" for base, _ in valid_pairs])
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.25)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linewidth=0.5)

    # Add legend to first subplot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_file = output_dir / f"optimization_impact_{version}{suffix}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    return fig


def plot_resource_limits_recommendation(
    results: dict,
    version: str,
    output_dir: Path,
    suffix: str = ""
) -> Optional[Figure]:
    """
    Create a chart showing recommended resource limits for K8s deployment.
    Visualizes peak vs average resource usage with safety margins.
    """
    version_data = results.get(version, {})
    if not version_data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Resource Limits Recommendation - Dataset {version}\n"
        f"(For Kubernetes Container Configuration)",
        fontsize=14, fontweight="bold"
    )

    engines = sorted(version_data.keys())
    x = range(len(engines))

    # Memory chart
    ax_mem = axes[0]
    peak_mem = []
    avg_mem = []

    for engine in engines:
        run = version_data[engine][-1]
        peak_mem.append(run.get("peak_memory_mb", 0))
        avg_mem.append(run.get("avg_memory_mb", 0))

    bar_width = 0.35
    bars1 = ax_mem.bar([i - bar_width/2 for i in x], avg_mem, bar_width,
                       label="Average", color="#3498db", alpha=0.7)
    bars2 = ax_mem.bar([i + bar_width/2 for i in x], peak_mem, bar_width,
                       label="Peak", color="#e74c3c", alpha=0.7)

    # Add recommended limit line (peak + 20% safety margin)
    for i, (engine, peak) in enumerate(zip(engines, peak_mem)):
        recommended = peak * 1.2
        ax_mem.hlines(recommended, i - 0.4, i + 0.4, colors="#2c3e50",
                      linestyles="--", linewidth=2)
        ax_mem.annotate(f"Limit: {recommended:.0f}MB",
                       xy=(i, recommended), xytext=(i, recommended + 20),
                       ha="center", fontsize=9, fontweight="bold",
                       color="#2c3e50")

    ax_mem.bar_label(bars1, fmt='%.0f', fontsize=8)
    ax_mem.bar_label(bars2, fmt='%.0f', fontsize=8)
    ax_mem.set_ylabel("Memory (MB)")
    ax_mem.set_title("Memory Usage & Recommended Limits", fontweight="bold")
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels([get_engine_style(e)["label"] for e in engines],
                           rotation=15, ha="right", fontsize=9)
    ax_mem.legend(loc="upper left")
    ax_mem.grid(True, alpha=0.3, axis="y")
    ax_mem.set_ylim(bottom=0, top=max(peak_mem) * 1.5)

    # CPU chart
    ax_cpu = axes[1]
    peak_cpu = []
    avg_cpu = []

    for engine in engines:
        run = version_data[engine][-1]
        peak_cpu.append(run.get("peak_cpu_percent", 0))
        avg_cpu.append(run.get("avg_cpu_percent", 0))

    bars1 = ax_cpu.bar([i - bar_width/2 for i in x], avg_cpu, bar_width,
                       label="Average", color="#3498db", alpha=0.7)
    bars2 = ax_cpu.bar([i + bar_width/2 for i in x], peak_cpu, bar_width,
                       label="Peak", color="#e74c3c", alpha=0.7)

    ax_cpu.bar_label(bars1, fmt='%.0f', fontsize=8)
    ax_cpu.bar_label(bars2, fmt='%.0f', fontsize=8)
    ax_cpu.set_ylabel("CPU Usage (%)")
    ax_cpu.set_title("CPU Usage Comparison", fontweight="bold")
    ax_cpu.set_xticks(x)
    ax_cpu.set_xticklabels([get_engine_style(e)["label"] for e in engines],
                           rotation=15, ha="right", fontsize=9)
    ax_cpu.legend(loc="upper left")
    ax_cpu.grid(True, alpha=0.3, axis="y")
    ax_cpu.axhline(y=100, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax_cpu.set_ylim(bottom=0, top=max(peak_cpu) * 1.3)

    plt.tight_layout()

    output_file = output_dir / f"resource_limits_{version}{suffix}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization charts from benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing benchmark results (default: results)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("charts"),
        help="Directory to save generated charts (default: charts)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specific dataset version to visualize (default: all)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display charts interactively"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="all",
        help=(
            "Engine configurations to compare (comma-separated). "
            "Options: native, native-optimized, pytesseract, pytesseract-optimized, all. "
            "Examples: --compare native,pytesseract or --compare native,native-optimized"
        )
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available engine configurations and exit"
    )

    args = parser.parse_args()

    # Load results first (needed for --list-engines)
    print(f"Loading results from: {args.results_dir}")
    results = load_benchmark_results(args.results_dir)

    if not results:
        print("No benchmark results found!", file=sys.stderr)
        sys.exit(1)

    # Collect all available engines across all versions
    all_engines = set()
    for version_data in results.values():
        all_engines.update(version_data.keys())
    all_engines = sorted(all_engines)

    # Handle --list-engines
    if args.list_engines:
        print(f"Available engine configurations: {', '.join(all_engines)}")
        sys.exit(0)

    # Parse --compare argument
    if args.compare.lower() == "all":
        engines_to_compare = None  # None means no filtering
    else:
        engines_to_compare = [e.strip() for e in args.compare.split(",")]
        # Validate engines
        invalid_engines = set(engines_to_compare) - set(all_engines)
        if invalid_engines:
            print(f"Error: Unknown engine(s): {', '.join(invalid_engines)}", file=sys.stderr)
            print(f"Available engines: {', '.join(all_engines)}", file=sys.stderr)
            sys.exit(1)

    # Filter results based on --compare
    results = filter_results_by_engines(results, engines_to_compare)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(results)} dataset version(s): {', '.join(sorted(results.keys()))}")
    if engines_to_compare:
        print(f"Comparing engines: {', '.join(engines_to_compare)}")
    else:
        print(f"Comparing all engines: {', '.join(all_engines)}")

    # Create suffix for output filenames based on comparison
    if engines_to_compare and len(engines_to_compare) < len(all_engines):
        suffix = "_" + "_vs_".join(sorted(engines_to_compare))
    else:
        suffix = ""

    # Generate charts
    versions_to_plot = [args.version] if args.version else sorted(results.keys())

    for version in versions_to_plot:
        if version not in results:
            print(f"Warning: Version {version} not found in results", file=sys.stderr)
            continue

        print(f"\nGenerating charts for version {version}...")
        plot_resource_timeline(results, version, args.output_dir, suffix)
        plot_pages_over_time(results, version, args.output_dir, suffix)
        plot_processing_time_per_page(results, version, args.output_dir, suffix)
        plot_page_level_metrics(results, version, args.output_dir, suffix)
        plot_optimization_impact(results, version, args.output_dir, suffix)
        plot_resource_limits_recommendation(results, version, args.output_dir, suffix)

    # Cross-version comparison (if multiple versions)
    if len(results) > 1:
        print("\nGenerating cross-version comparison...")
        plot_comparison_across_versions(results, args.output_dir, suffix)

    print(f"\nAll charts saved to: {args.output_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
