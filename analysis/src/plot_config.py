"""Shared per-model colors and matplotlib defaults used across the figures.

Vendor color scheme (from the user, 2026-05-02):
  gpt       - teal/green       (#00A67E, ChatGPT mark)
  claude    - rust/burnt orange (#C15F3C)
  glm       - grey             (monochrome family)
  qwen      - purple           (#5B43D4, "Purple Heart")
  deepseek  - bright blue      (#2B6CB4)
  kimi      - black/monochrome (#1A1A1A)

For multiple variants from the same vendor (e.g. gpt-5.4 vs gpt-5.5,
opus-4.6 vs opus-4.7), the newest/strongest model uses the base color and
older variants use a lighter or darker shade.

Importing this module also installs the standard matplotlib style for the
analysis figures (no top/right spines, no grid lines).
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage
from matplotlib.patches import Circle
import scienceplots  # noqa: F401  (registers matplotlib styles)

# ---------------------------------------------------------------------------
# Canonical figure style. EVERY plot script in analysis/src/ should pick up
# this style by importing `plot_config` (even just `from plot_config import
# style_axes`). The two settings any new chart MUST inherit:
#   - SCIENCE_STYLES: scienceplots base sheet (without LaTeX deps)
#   - FONT_FAMILY:    DejaVu Serif everywhere for consistent local rendering
# Add new defaults to RC_DEFAULTS so they apply globally.
# ---------------------------------------------------------------------------

SCIENCE_STYLES: list[str] = ["science", "no-latex"]
FONT_FAMILY: str = "DejaVu Serif"

RC_DEFAULTS: dict = {
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 16,
    "font.family": FONT_FAMILY,
    "mathtext.fontset": "cm",
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 2.4,
}


def apply_style() -> None:
    """Idempotently install the project-wide matplotlib style.

    Called once at import time; safe to re-call from any script that wants
    to be defensive about other code (e.g. seaborn) clobbering the rcParams.
    """
    plt.style.use(SCIENCE_STYLES)
    plt.rcParams.update(RC_DEFAULTS)


apply_style()


def style_axes(ax) -> None:
    """Apply per-axes hardening for spines + grid (rcParams sometimes lose to styles)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


VENDOR_BASE: dict[str, str] = {
    "gpt":      "#00A67E",
    "claude":   "#C15F3C",
    "glm":      "#6E6E6E",
    "qwen":     "#5B43D4",
    "deepseek": "#2B6CB4",
    "kimi":     "#1A1A1A",
}

# Per-model overrides. Keys are the clean model name that appears in the legend
# *before* the scaffold parenthesis (e.g. "gpt-5.5", "glm-5.1").
MODEL_COLORS: dict[str, str] = {
    # gpt
    "gpt-5.5":           "#00A67E",
    "gpt-5.4":           "#4FBF9A",  # lighter mint for the older variant
    "gpt-5.3-spark":     "#7FD9B9",
    # claude
    "claude-opus-4.6":   "#C15F3C",
    "claude-opus-4.7":   "#8E4124",
    "claude-sonnet-4.6": "#E0866A",
    # glm (grey family — kept distinct from kimi's near-black)
    "glm-5.1":           "#6E6E6E",
    "glm-5":             "#8C8C8C",
    "glm-4.7":           "#B0B0B0",
    # qwen
    "qwen-3.6-plus":     "#5B43D4",
    # deepseek
    "deepseek-v4-pro":   "#2B6CB4",
    "deepseek-v4-flash": "#6BA4D6",
    # kimi
    "kimi-k2.6":         "#1A1A1A",
}


def color_for_model(model: str) -> str:
    """Return the color for a clean model name (e.g. 'gpt-5.5', 'glm-5.1')."""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    for prefix, color in VENDOR_BASE.items():
        if model.startswith(prefix):
            return color
    return "#444444"


def color_for_label(label: str) -> str:
    """Return the color for a 'model (Scaffold)' style legend label."""
    model = label.split(" (")[0]
    return color_for_model(model)


# ---- vendor + logo support ---------------------------------------------------

_VENDOR_PREFIXES = ("gpt", "claude", "glm", "qwen", "deepseek", "kimi")

# Where vendor logo PNGs live. Drop one PNG per vendor at these paths and they
# get rendered next to each line automatically. Missing files are silently
# skipped (with a one-time warning printed by `report_missing_logos`).
LOGO_DIR = Path(__file__).resolve().parents[1] / "assets" / "logos"
LOGO_FILENAMES: dict[str, str] = {
    "gpt":      "openai.png",
    "claude":   "anthropic.png",
    "glm":      "zhipu.png",
    "qwen":     "qwen.png",
    "deepseek": "deepseek.png",
    "kimi":     "moonshot.png",
}


def vendor_for_model(model: str) -> str | None:
    for v in _VENDOR_PREFIXES:
        if model.startswith(v):
            return v
    return None


def vendor_for_label(label: str) -> str | None:
    return vendor_for_model(label.split(" (")[0])


def logo_path_for_label(label: str) -> Path | None:
    v = vendor_for_label(label)
    if v is None:
        return None
    fn = LOGO_FILENAMES.get(v)
    if not fn:
        return None
    return LOGO_DIR / fn


def place_model_logo(ax, x, y, label: str, zoom: float = 0.06,
                     x_offset_pts: float = 14.0,
                     y_offset_pts: float = 0.0,
                     zorder: float = 5.0) -> bool:
    """Place a small vendor logo at data point (x, y), offset in display points.

    Returns True if a logo was actually drawn, False if the file is missing.
    """
    p = logo_path_for_label(label)
    if p is None or not p.is_file():
        return False
    img = mpimg.imread(p)
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        im, (x, y),
        xybox=(x_offset_pts, y_offset_pts),
        xycoords="data",
        boxcoords="offset points",
        frameon=False,
        pad=0,
        zorder=zorder,
    )
    ax.add_artist(ab)
    return True


def _place_badge(ax, x, y, color, radius_pts: float,
                 x_offset_pts: float, y_offset_pts: float,
                 zorder: float = 4.0,
                 linestyle: str = "-") -> None:
    """Draw a colored ring (white fill, colored stroke) at offset position.

    Pass ``linestyle="--"`` for a dashed ring (used to mark active-memory line
    endpoints so the badge style mirrors the line style).
    """
    diam = radius_pts * 2
    da = DrawingArea(diam, diam, 0, 0)
    da.add_artist(
        Circle((radius_pts, radius_pts), radius_pts,
               facecolor="white", edgecolor=color, linewidth=2.0,
               linestyle=linestyle)
    )
    ab = AnnotationBbox(
        da, (x, y),
        xybox=(x_offset_pts, y_offset_pts),
        xycoords="data",
        boxcoords="offset points",
        frameon=False,
        pad=0,
        zorder=zorder,
    )
    ax.add_artist(ab)


def place_endcap(ax, x, y, label: str, value_str: str,
                 zoom: float = 0.05,
                 badge_radius_pts: float = 12.0,
                 y_offset_pts: float = 0.0,
                 x_offset_pts: float = 0.0,
                 name_above: bool = False,
                 show_name: bool = True,
                 show_badge: bool = True,
                 linestyle: str = "-") -> None:
    """KellyBench-style end-of-line marker.

    The line endpoint sits at the centre of a colored ring containing the
    vendor logo. The value text sits immediately to the right of the ring.
    The model name is centred above the ring (when ``name_above=True``) or
    centred below the ring (default). Per-model `name_above` lets adjacent
    endcaps put their name on opposite sides of the badge so the labels
    don't crash into each other.

    Set ``show_name=False`` to suppress the model name (e.g. when the legend
    already labels the colors). Set ``show_badge=False`` to suppress the
    colored ring + vendor logo (keeping just the value text). Pass
    ``linestyle="--"`` to draw a dashed ring, mirroring a dashed line endpoint.
    """
    color = color_for_label(label)
    model_name = label.split(" (")[0]
    if show_badge:
        _place_badge(ax, x, y, color, badge_radius_pts,
                     x_offset_pts=x_offset_pts, y_offset_pts=y_offset_pts,
                     linestyle=linestyle)
        place_model_logo(ax, x, y, label, zoom=zoom,
                         x_offset_pts=x_offset_pts, y_offset_pts=y_offset_pts)
    if show_name:
        if name_above:
            name_y = y_offset_pts + badge_radius_pts + 4.0
            name_va = "bottom"
        else:
            name_y = y_offset_pts - badge_radius_pts - 4.0
            name_va = "top"
        ax.annotate(
            model_name,
            xy=(x, y),
            xycoords="data",
            xytext=(x_offset_pts, name_y),
            textcoords="offset points",
            ha="center",
            va=name_va,
            color=color,
            fontsize=11,
            fontweight="bold",
            annotation_clip=False,
            zorder=10,  # draw on top of any neighbouring badge
            # White halo so the name stays legible if it overlaps a neighbour badge.
            path_effects=[pe.withStroke(linewidth=4.0, foreground="white")],
        )
    # When the badge is hidden, the value text sits right at the line
    # endpoint with a small gap; otherwise it sits to the right of the badge.
    value_x_pad = (badge_radius_pts + 4.0) if show_badge else 4.0
    ax.annotate(
        value_str,
        xy=(x, y),
        xycoords="data",
        xytext=(x_offset_pts + value_x_pad, y_offset_pts),
        textcoords="offset points",
        ha="left",
        va="center",
        color=color,
        fontsize=12,
        fontweight="bold",
        annotation_clip=False,
    )


def _tangent_offsets(ax, endpoints, radius_pts: float = 12.0):
    """Per-endpoint y-offset (display points) so overlapping badges become
    tangent to their line endpoint instead of intersecting.

    Endpoints are tuples ``(label, x, y_data, value_str)``. We convert ``y_data``
    to display pixels, sort ascending, and group consecutive pairs whose pixel
    gap is less than one badge diameter into clusters. For each cluster of
    size >=2:

      - lowest badge gets ``-radius_pts`` (badge sits below line, top edge of
        the badge touches the line endpoint)
      - highest badge gets ``+radius_pts`` (badge sits above line, bottom edge
        of the badge touches the line endpoint)
      - any middle members stay at offset 0 (rare; if a 3-way crash happens in
        practice, override that single badge manually).

    The center-to-center distance after the shift is ``(y_upper - y_lower) +
    2*radius_pts``, so when the two endpoints share the same y the badges end
    up exactly touching, and any natural separation widens the gap further.
    """
    if not endpoints:
        return []
    fig_dpi = ax.figure.dpi
    pts_per_pixel = 72.0 / fig_dpi
    pixel_ys = [ax.transData.transform((0.0, e[2]))[1] for e in endpoints]
    diameter_px = 2 * radius_pts / pts_per_pixel
    order = sorted(range(len(endpoints)), key=lambda i: pixel_ys[i])
    sorted_pys = [pixel_ys[i] for i in order]
    clusters: list[list[int]] = [[order[0]]]
    for k in range(1, len(order)):
        if sorted_pys[k] - sorted_pys[k - 1] < diameter_px:
            clusters[-1].append(order[k])
        else:
            clusters.append([order[k]])
    offsets = [0.0] * len(endpoints)
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        offsets[cluster[0]] = -radius_pts
        offsets[cluster[-1]] = +radius_pts
    return offsets


def _resolve_endpoint_offsets(ax, endpoints, min_gap_pts: float = 12.0):
    """Return per-endpoint y-offsets (in *display points*) so no two labels sit
    closer than `min_gap_pts` vertically. Offsets are returned in input order
    and can be passed directly to `place_endcap(..., y_offset_pts=...)`.
    """
    if not endpoints:
        return []
    fig_dpi = ax.figure.dpi
    pts_per_pixel = 72.0 / fig_dpi  # convert pixel deltas back to display points
    # transData returns coordinates in display *pixels*.
    pixel_ys = [ax.transData.transform((0.0, e[2]))[1] for e in endpoints]
    order = sorted(range(len(endpoints)), key=lambda i: pixel_ys[i])
    desired = [pixel_ys[i] for i in order]
    adjusted = list(desired)
    min_gap_pixels = min_gap_pts / pts_per_pixel
    for k in range(1, len(adjusted)):
        if adjusted[k] - adjusted[k - 1] < min_gap_pixels:
            adjusted[k] = adjusted[k - 1] + min_gap_pixels
    offsets_pts = [0.0] * len(endpoints)
    for k, idx in enumerate(order):
        offsets_pts[idx] = (adjusted[k] - desired[k]) * pts_per_pixel
    return offsets_pts


def format_metric_value(metric: str, value: float) -> str:
    if metric == "accuracy":
        return f"{value:.0f}%"
    if metric == "avg_brier":
        return f"{value:.2f}"
    return f"{value:.2f}"


def report_missing_logos(labels) -> None:
    """Print one warning listing vendors whose logo PNG is missing."""
    needed: dict[str, Path] = {}
    for lbl in labels:
        v = vendor_for_label(lbl)
        if v is None:
            continue
        p = logo_path_for_label(lbl)
        if p is not None and not p.is_file():
            needed[v] = p
    if not needed:
        return
    print(f"  [logos] missing — drop PNGs at:")
    for v, p in needed.items():
        print(f"    {v:<10s} -> {p}")
