from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import mne


# ---------------------------
# Channel presets
# ---------------------------

CLINICAL_19 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

POSTERIOR = ["P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
FRONTAL = ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "F3", "F4", "Fz"]


def _norm_name(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def pick_channels_by_name(raw: mne.io.BaseRaw, wanted: Sequence[str]) -> List[str]:
    name_map = {_norm_name(ch): ch for ch in raw.ch_names}
    out = []
    for w in wanted:
        key = _norm_name(w)
        if key in name_map:
            out.append(name_map[key])
    return out


def choose_channels(raw: mne.io.BaseRaw, mode: str, custom: Optional[List[str]]) -> List[str]:
    if custom:
        picked = pick_channels_by_name(raw, custom)
        if len(picked) < 4:
            raise RuntimeError(f"Too few custom channels matched. Got {picked}")
        return picked

    mode = mode.lower()
    if mode == "clinical_19":
        picked = pick_channels_by_name(raw, CLINICAL_19)
        return picked if len(picked) >= 8 else raw.ch_names[: min(32, len(raw.ch_names))]
    if mode == "posterior":
        picked = pick_channels_by_name(raw, POSTERIOR)
        return picked if len(picked) >= 4 else pick_channels_by_name(raw, CLINICAL_19)
    if mode == "frontal":
        picked = pick_channels_by_name(raw, FRONTAL)
        return picked if len(picked) >= 4 else pick_channels_by_name(raw, CLINICAL_19)
    if mode == "first_32":
        return raw.ch_names[: min(32, len(raw.ch_names))]
    raise ValueError(f"Unknown channel mode: {mode}")


# ---------------------------
# Predictions and Targets
# ---------------------------

@dataclass
class PredInterval:
    t_start: float
    t_end: float
    pred: int
    prob: Optional[float] = None
    target: Optional[int] = None


def load_intervals(path: str) -> List[PredInterval]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in [".json"]:
        obj = json.loads(p.read_text(encoding="utf-8"))
        out: List[PredInterval] = []
        for it in obj:
            out.append(PredInterval(
                t_start=float(it["t_start"]),
                t_end=float(it["t_end"]),
                pred=int(it["pred"]),
                prob=float(it["prob"]) if "prob" in it and it["prob"] is not None else None,
                target=int(it["target"]) if "target" in it and it["target"] is not None else None,
            ))
        return out

    if p.suffix.lower() in [".csv", ".tsv"]:
        out: List[PredInterval] = []
        delim = "\t" if p.suffix.lower() == ".tsv" else ","
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter=delim)
            for row in r:
                out.append(PredInterval(
                    t_start=float(row["t_start"]),
                    t_end=float(row["t_end"]),
                    pred=int(row["pred"]),
                    prob=float(row["prob"]) if "prob" in row and row["prob"] != "" else None,
                    target=int(row["target"]) if "target" in row and row["target"] != "" else None,
                ))
        return out

    raise ValueError("Intervals must be .json or .csv/.tsv with columns t_start,t_end,pred[,prob]")


# ---------------------------
# Color mapping for classes
# ---------------------------

def get_class_color(class_id: int, class_labels: Dict[int, str], color_map: Optional[Dict[int, str]] = None) -> str:
    """Get color for a class. Uses color_map if provided, otherwise uses default colors."""
    if color_map and class_id in color_map:
        return color_map[class_id]
    
    # Default color scheme
    default_colors = {
        0: "#4C72B0",  # blue for closed
        1: "#DD8452",  # orange for opened
        2: "#55A868",  # green
        3: "#C44E52",  # red
        4: "#8172B3",  # purple
        5: "#CCB974",  # yellow
        6: "#64B5CD",  # cyan
        7: "#DA8BC3",  # pink
    }
    
    if class_id in default_colors:
        return default_colors[class_id]
    
    # Generate a color for unknown classes using a colormap
    n = len(default_colors)
    if class_id < n:
        return default_colors[class_id]
    # Use a hash-based color for classes beyond defaults
    import hashlib
    h = int(hashlib.md5(str(class_id).encode()).hexdigest()[:6], 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------
# Plot config
# ---------------------------

@dataclass
class PlotCfg:
    set_path: str
    intervals_path: str
    out_pdf: str
    class_labels: Optional[Dict[int, str]] = None
    class_colors: Optional[Dict[int, str]] = None

    # Display preprocessing (plot-only)
    l_freq: Optional[float] = 0.5
    h_freq: Optional[float] = 40.0
    notch: Optional[float] = 50.0   # set 60 if needed, or None

    # Resample for plotting (smaller PDF + faster)
    plot_sfreq: float = 125.0

    # Channels
    channel_mode: str = "clinical_19"
    custom_channels: Optional[List[str]] = None

    # How much time per panel, how many panels per page
    panel_sec: float = 30.0
    panels_per_page: int = 4   # 1, 2, or 4

    # Visual scale
    scale_uV: float = 80.0
    linewidth: float = 0.7

    # Grid
    major_sec: float = 10.0
    minor_sec: float = 2.0

    # Overlays
    pred_alpha: float = 0.12
    ann_alpha: float = 0.06
    show_pred_bar: bool = True  # for 4 panels it can look busy; you can set False

    # PDF weight
    pdf_dpi: int = 150
    rasterize_traces: bool = True

    def __post_init__(self):
        if self.class_labels is None:
            self.class_labels = {0: "closed", 1: "opened"}


def format_mmss(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def intersect(a0: float, a1: float, b0: float, b1: float) -> Optional[Tuple[float, float]]:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return (lo, hi) if hi > lo else None


def draw_time_grid(ax, t0: float, t1: float, cfg: PlotCfg) -> None:
    major = np.arange(math.floor(t0 / cfg.major_sec) * cfg.major_sec, t1 + 1e-9, cfg.major_sec)
    minor = np.arange(math.floor(t0 / cfg.minor_sec) * cfg.minor_sec, t1 + 1e-9, cfg.minor_sec)

    ax.set_xticks(major)
    ax.set_xticks(minor, minor=True)

    ax.grid(which="major", axis="x", alpha=0.35, linewidth=0.8)
    ax.grid(which="minor", axis="x", alpha=0.18, linewidth=0.6)
    ax.grid(which="major", axis="y", alpha=0.10, linewidth=0.6)


def load_and_prepare_raw(set_path: str, l_freq: Optional[float], h_freq: Optional[float], 
                         notch: Optional[float], plot_sfreq: float) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")

    # Normalize channel names a bit (spaces etc.)
    raw.rename_channels(lambda s: s.strip())

    # Keep EEG-ish channels (if types are present)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False, stim=False, misc=True)
    raw.pick(picks)

    # Reference for readability
    try:
        raw.set_eeg_reference("average", projection=False)
    except Exception:
        pass

    # Filters for display
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")

    if notch is not None:
        raw.notch_filter([notch], verbose="ERROR")

    # Resample for plotting
    sf = float(raw.info["sfreq"])
    if plot_sfreq is not None and plot_sfreq < sf:
        raw.resample(plot_sfreq, npad="auto", verbose="ERROR")

    return raw


def plot_one_panel(
    fig: plt.Figure,
    raw: mne.io.BaseRaw,
    intervals: List[PredInterval],
    target_intervals: Optional[List[PredInterval]],
    ch_names: List[str],
    cfg: PlotCfg,
    t0: float,
    t1: float,
    gs_slot,
    add_legend: bool = False,
) -> None:
    # Layout: traces + prediction bar + (optionally) target bar
    has_targets = target_intervals is not None and len(target_intervals) > 0
    if cfg.show_pred_bar:
        if has_targets:
            sub = gs_slot.subgridspec(3, 1, height_ratios=[14, 1.0, 1.0], hspace=0.05)
            ax = fig.add_subplot(sub[0])
            axp = fig.add_subplot(sub[1], sharex=ax)
            axt = fig.add_subplot(sub[2], sharex=ax)
        else:
            sub = gs_slot.subgridspec(2, 1, height_ratios=[14, 1.0], hspace=0.05)
            ax = fig.add_subplot(sub[0])
            axp = fig.add_subplot(sub[1], sharex=ax)
            axt = None
    else:
        ax = fig.add_subplot(gs_slot)
        axp = None
        axt = None

    sf = float(raw.info["sfreq"])
    start = int(round(t0 * sf))
    stop = int(round(t1 * sf))
    times = np.arange(stop - start) / sf + t0

    picks = [raw.ch_names.index(ch) for ch in ch_names]
    data = raw.get_data(picks=picks, start=start, stop=stop) * 1e6  # uV

    n_ch = data.shape[0]
    offsets = np.arange(n_ch)[::-1].astype(float)
    y = data / cfg.scale_uV + offsets[:, None]

    # Background spans: predictions
    for it in intervals:
        inter = intersect(t0, t1, it.t_start, it.t_end)
        if inter is None:
            continue
        x0, x1 = inter
        color = get_class_color(it.pred, cfg.class_labels, cfg.class_colors)
        ax.axvspan(x0, x1, color=color, alpha=cfg.pred_alpha, linewidth=0)
        if axp is not None:
            axp.axvspan(x0, x1, color=color, alpha=0.85, linewidth=0)

    # Background spans: targets (if provided)
    if target_intervals is not None:
        for it in target_intervals:
            inter = intersect(t0, t1, it.t_start, it.t_end)
            if inter is None:
                continue
            x0, x1 = inter
            color = get_class_color(it.pred, cfg.class_labels, cfg.class_colors)
            if axt is not None:
                axt.axvspan(x0, x1, color=color, alpha=0.85, linewidth=0)

    # Background spans: annotations (if durations exist)
    if raw.annotations is not None and len(raw.annotations) > 0:
        for an in raw.annotations:
            a0 = float(an["onset"])
            a1 = float(an["onset"] + an["duration"])
            inter = intersect(t0, t1, a0, a1)
            if inter is None:
                continue
            x0, x1 = inter
            ax.axvspan(x0, x1, alpha=cfg.ann_alpha)

    # Annotation onset markers (thin lines)
    if raw.annotations is not None and len(raw.annotations) > 0:
        for an in raw.annotations:
            x = float(an["onset"])
            if t0 <= x <= t1:
                ax.axvline(x, alpha=0.45, linewidth=1.0)
                if axp is not None:
                    axp.axvline(x, alpha=0.45, linewidth=1.0)
                if axt is not None:
                    axt.axvline(x, alpha=0.45, linewidth=1.0)

    # Traces
    for i in range(n_ch):
        ln, = ax.plot(times, y[i], linewidth=cfg.linewidth)
        ln.set_antialiased(True)
        ln.set_solid_capstyle("butt")   # чёткие окончания
        ln.set_solid_joinstyle("miter") # чёткие стыки
        if cfg.rasterize_traces:
            ln.set_rasterized(True)

    # Axis formatting
    ax.set_xlim(t0, t1)
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names, fontsize=9)
    draw_time_grid(ax, t0, t1, cfg)
    
    # Add legend if requested
    if add_legend:
        create_legend(ax, cfg)

    # x ticks labels in mm:ss (only on bottom axis)
    if axp is not None:
        plt.setp(ax.get_xticklabels(), visible=False)
        axp.set_ylim(0, 1)
        axp.set_yticks([])
        axp.set_ylabel("pred", rotation=0, labelpad=18)
        draw_time_grid(axp, t0, t1, cfg)
        
        if axt is not None:
            plt.setp(axp.get_xticklabels(), visible=False)
            axt.set_ylim(0, 1)
            axt.set_yticks([])
            axt.set_ylabel("target", rotation=0, labelpad=18)
            draw_time_grid(axt, t0, t1, cfg)
            
            major = axt.get_xticks()
            axt.set_xticklabels([format_mmss(x) for x in major], fontsize=9)
            axt.set_xlabel("Time (mm:ss)")
        else:
            major = axp.get_xticks()
            axp.set_xticklabels([format_mmss(x) for x in major], fontsize=9)
            axp.set_xlabel("Time (mm:ss)")
    else:
        major = ax.get_xticks()
        ax.set_xticklabels([format_mmss(x) for x in major], fontsize=9)
        ax.set_xlabel("Time (mm:ss)")


def build_page_windows(total_sec: float, panel_sec: float, panels_per_page: int) -> List[List[Tuple[float, float]]]:
    page_span = panel_sec * panels_per_page
    pages: List[List[Tuple[float, float]]] = []

    t = 0.0
    while t < total_sec - 1e-9:
        page = []
        for _ in range(panels_per_page):
            a = t
            b = min(t + panel_sec, total_sec)
            if b <= a:
                break
            page.append((a, b))
            t = b
        pages.append(page)
    return pages


def create_legend(ax, cfg: PlotCfg) -> None:
    """Create a legend showing class labels with colors on the given axis."""
    if cfg.class_labels is None or len(cfg.class_labels) == 0:
        return
    
    # Get unique classes from labels
    classes = sorted(cfg.class_labels.keys())
    legend_elements = []
    
    for class_id in classes:
        label = cfg.class_labels[class_id]
        color = get_class_color(class_id, cfg.class_labels, cfg.class_colors)
        patch = Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=f"{class_id}: {label}")
        legend_elements.append(patch)
    
    # Add legend to the axis
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, ncol=min(len(classes), 4))


def plot_eeg_with_predictions(
    set_path: str,
    intervals_path: str,
    out_pdf: str,
    class_labels: Optional[Dict[int, str]] = None,
    class_colors: Optional[Dict[int, str]] = None,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 40.0,
    notch: Optional[float] = 50.0,
    plot_sfreq: float = 125.0,
    channel_mode: str = "clinical_19",
    custom_channels: Optional[List[str]] = None,
    panel_sec: float = 30.0,
    panels_per_page: int = 4,
    scale_uV: float = 80.0,
    linewidth: float = 0.7,
    major_sec: float = 10.0,
    minor_sec: float = 2.0,
    pred_alpha: float = 0.12,
    ann_alpha: float = 0.06,
    show_pred_bar: bool = True,
    pdf_dpi: int = 150,
    rasterize_traces: bool = True,
) -> None:
    """
    Plot EEG data with model predictions and optionally target labels.
    
    Args:
        set_path: Path to EEGLAB .set file (required)
        intervals_path: Path to intervals JSON/CSV file with pred and optionally target fields (required)
        out_pdf: Output PDF path (required)
        class_labels: Dictionary mapping class IDs to labels (default: {0: "closed", 1: "opened"})
        class_colors: Optional dictionary mapping class IDs to hex colors (overrides defaults)
        l_freq: High-pass filter frequency (Hz)
        h_freq: Low-pass filter frequency (Hz)
        notch: Notch filter frequency (Hz), set None to disable
        plot_sfreq: Resampling frequency for plotting (Hz)
        channel_mode: Channel selection mode ("clinical_19", "posterior", "frontal", "first_32")
        custom_channels: List of custom channel names (overrides channel_mode)
        panel_sec: Seconds per panel
        panels_per_page: Number of panels per page (1, 2, or 4)
        scale_uV: Microvolts per vertical unit (larger = smaller traces)
        linewidth: Line width for EEG traces
        major_sec: Major grid spacing (seconds)
        minor_sec: Minor grid spacing (seconds)
        pred_alpha: Alpha transparency for prediction overlays
        ann_alpha: Alpha transparency for annotation overlays
        show_pred_bar: Whether to show prediction/target bars
        pdf_dpi: DPI for PDF output
        rasterize_traces: Whether to rasterize traces (smaller PDF size)
    """
    if class_labels is None:
        class_labels = {0: "closed", 1: "opened"}
    
    cfg = PlotCfg(
        set_path=set_path,
        intervals_path=intervals_path,
        out_pdf=out_pdf,
        class_labels=class_labels,
        class_colors=class_colors,
        l_freq=l_freq,
        h_freq=h_freq,
        notch=notch,
        plot_sfreq=plot_sfreq,
        channel_mode=channel_mode,
        custom_channels=custom_channels,
        panel_sec=panel_sec,
        panels_per_page=panels_per_page,
        scale_uV=scale_uV,
        linewidth=linewidth,
        major_sec=major_sec,
        minor_sec=minor_sec,
        pred_alpha=pred_alpha,
        ann_alpha=ann_alpha,
        show_pred_bar=show_pred_bar,
        pdf_dpi=pdf_dpi,
        rasterize_traces=rasterize_traces,
    )
    
    raw = load_and_prepare_raw(cfg.set_path, cfg.l_freq, cfg.h_freq, cfg.notch, cfg.plot_sfreq)
    intervals = load_intervals(cfg.intervals_path)
    
    # Create target_intervals from the same data, using target field
    target_intervals = None
    if intervals:
        # Create target intervals from intervals that have target field (target can be 0, so we check is not None)
        target_intervals_list = [
            PredInterval(t_start=it.t_start, t_end=it.t_end, pred=it.target, prob=it.prob)
            for it in intervals if it.target is not None
        ]
        if target_intervals_list:
            target_intervals = target_intervals_list

    # pick channels
    ch_names = choose_channels(raw, cfg.channel_mode, cfg.custom_channels)

    total_sec = raw.n_times / float(raw.info["sfreq"])

    # build page windows
    if cfg.panels_per_page not in (1, 2, 4):
        raise ValueError("panels_per_page must be 1, 2, or 4")

    pages = build_page_windows(total_sec, cfg.panel_sec, cfg.panels_per_page)

    out_path = Path(cfg.out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_title = (
        f"{Path(cfg.set_path).name} | sf={raw.info['sfreq']:.1f}Hz | "
        f"ch={len(ch_names)} | scale={cfg.scale_uV:.0f}µV/unit | "
        f"panel={cfg.panel_sec:.0f}s x {cfg.panels_per_page} per page"
    )

    with PdfPages(out_path) as pdf:
        for pi, page in enumerate(pages, start=1):
            # Figure layout per page
            if cfg.panels_per_page == 1:
                fig = plt.figure(figsize=(14, 8))
                gs = fig.add_gridspec(1, 1)
                plot_one_panel(fig, raw, intervals, target_intervals, ch_names, cfg, page[0][0], page[0][1], gs[0, 0], add_legend=(pi == 1))

            elif cfg.panels_per_page == 2:
                fig = plt.figure(figsize=(14, 10))
                gs = fig.add_gridspec(2, 1, hspace=0.18)
                plot_one_panel(fig, raw, intervals, target_intervals, ch_names, cfg, page[0][0], page[0][1], gs[0, 0], add_legend=(pi == 1))
                if len(page) > 1:
                    plot_one_panel(fig, raw, intervals, target_intervals, ch_names, cfg, page[1][0], page[1][1], gs[1, 0])

            else:  # 4 panels
                fig = plt.figure(figsize=(16, 10))
                gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.10)
                slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for k, (a, b) in enumerate(page):
                    r, c = slots[k]
                    plot_one_panel(fig, raw, intervals, target_intervals, ch_names, cfg, a, b, gs[r, c], add_legend=(pi == 1 and k == 0))

            fig.suptitle(f"EEG + model predictions | {base_title} | page {pi}/{len(pages)}", fontsize=13)
            pdf.savefig(fig, dpi=cfg.pdf_dpi, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {out_path}")
