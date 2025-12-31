from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
# Predictions
# ---------------------------

@dataclass
class PredInterval:
    t_start: float
    t_end: float
    pred: int
    prob: Optional[float] = None


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
                ))
        return out

    raise ValueError("Intervals must be .json or .csv/.tsv with columns t_start,t_end,pred[,prob]")


# ---------------------------
# Plot config
# ---------------------------

@dataclass
class PlotCfg:
    set_path: str
    intervals_path: str
    out_pdf: str

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


def load_and_prepare_raw(cfg: PlotCfg) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(cfg.set_path, preload=True, verbose="ERROR")

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
    if cfg.l_freq is not None or cfg.h_freq is not None:
        raw.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin", verbose="ERROR")

    if cfg.notch is not None:
        raw.notch_filter([cfg.notch], verbose="ERROR")

    # Resample for plotting
    sf = float(raw.info["sfreq"])
    if cfg.plot_sfreq is not None and cfg.plot_sfreq < sf:
        raw.resample(cfg.plot_sfreq, npad="auto", verbose="ERROR")

    return raw


def plot_one_panel(
    fig: plt.Figure,
    raw: mne.io.BaseRaw,
    intervals: List[PredInterval],
    ch_names: List[str],
    cfg: PlotCfg,
    t0: float,
    t1: float,
    gs_slot,
) -> None:
    # optionally two-row layout: traces + prediction bar
    if cfg.show_pred_bar:
        sub = gs_slot.subgridspec(2, 1, height_ratios=[14, 1.6], hspace=0.05)
        ax = fig.add_subplot(sub[0])
        axp = fig.add_subplot(sub[1], sharex=ax)
    else:
        ax = fig.add_subplot(gs_slot)
        axp = None

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
        color = "#DD8452" if it.pred == 1 else "#4C72B0"
        ax.axvspan(x0, x1, color=color, alpha=cfg.pred_alpha, linewidth=0)
        if axp is not None:
            axp.axvspan(x0, x1, color=color, alpha=0.85, linewidth=0)

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

    # x ticks labels in mm:ss (only on bottom axis)
    if axp is not None:
        plt.setp(ax.get_xticklabels(), visible=False)
        axp.set_ylim(0, 1)
        axp.set_yticks([])
        axp.set_ylabel("pred", rotation=0, labelpad=18)
        draw_time_grid(axp, t0, t1, cfg)

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


def main(cfg: PlotCfg) -> None:
    raw = load_and_prepare_raw(cfg)
    intervals = load_intervals(cfg.intervals_path)

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
                plot_one_panel(fig, raw, intervals, ch_names, cfg, page[0][0], page[0][1], gs[0, 0])

            elif cfg.panels_per_page == 2:
                fig = plt.figure(figsize=(14, 10))
                gs = fig.add_gridspec(2, 1, hspace=0.18)
                plot_one_panel(fig, raw, intervals, ch_names, cfg, page[0][0], page[0][1], gs[0, 0])
                if len(page) > 1:
                    plot_one_panel(fig, raw, intervals, ch_names, cfg, page[1][0], page[1][1], gs[1, 0])

            else:  # 4 panels
                fig = plt.figure(figsize=(16, 10))
                gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.10)
                slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for k, (a, b) in enumerate(page):
                    r, c = slots[k]
                    plot_one_panel(fig, raw, intervals, ch_names, cfg, a, b, gs[r, c])

            fig.suptitle(f"EEG + model predictions | {base_title} | page {pi}/{len(pages)}", fontsize=13)
            pdf.savefig(fig, dpi=cfg.pdf_dpi, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {out_path}")


def parse_args() -> PlotCfg:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_path", required=True, help="Path to EEGLAB .set")
    ap.add_argument("--intervals", dest="intervals_path", required=True, help="JSON/CSV intervals: t_start,t_end,pred[,prob]")
    ap.add_argument("--out", dest="out_pdf", default="eeg_with_predictions.pdf", help="Output PDF path")

    ap.add_argument("--plot-sfreq", type=float, default=125.0, help="Resample for plotting (Hz)")
    ap.add_argument("--l-freq", type=float, default=0.5, help="High-pass (Hz)")
    ap.add_argument("--h-freq", type=float, default=40.0, help="Low-pass (Hz)")
    ap.add_argument("--notch", type=float, default=50.0, help="Notch (Hz), set -1 to disable")

    ap.add_argument("--channel-mode", default="clinical_19", choices=["clinical_19", "posterior", "frontal", "first_32"])
    ap.add_argument("--channels", default="", help="Custom channels comma-separated (overrides mode)")

    ap.add_argument("--panel-sec", type=float, default=30.0, help="Seconds per panel")
    ap.add_argument("--panels-per-page", type=int, default=4, choices=[1, 2, 4])

    ap.add_argument("--scale-uv", type=float, default=80.0, help="uV per vertical unit (bigger -> traces smaller)")
    ap.add_argument("--linewidth", type=float, default=0.7)

    ap.add_argument("--major-sec", type=float, default=10.0)
    ap.add_argument("--minor-sec", type=float, default=2.0)

    ap.add_argument("--pred-alpha", type=float, default=0.12)
    ap.add_argument("--ann-alpha", type=float, default=0.06)

    ap.add_argument("--no-pred-bar", action="store_true", help="Do not draw bottom prediction bar")
    ap.add_argument("--pdf-dpi", type=int, default=150)
    ap.add_argument("--no-raster", action="store_true", help="Do not rasterize traces (PDF might be heavier)")

    ns = ap.parse_args()

    notch = None if ns.notch < 0 else float(ns.notch)
    custom = [c.strip() for c in ns.channels.split(",") if c.strip()] or None

    return PlotCfg(
        set_path=ns.set_path,
        intervals_path=ns.intervals_path,
        out_pdf=ns.out_pdf,
        l_freq=float(ns.l_freq) if ns.l_freq is not None else None,
        h_freq=float(ns.h_freq) if ns.h_freq is not None else None,
        notch=notch,
        plot_sfreq=float(ns.plot_sfreq),
        channel_mode=ns.channel_mode,
        custom_channels=custom,
        panel_sec=float(ns.panel_sec),
        panels_per_page=int(ns.panels_per_page),
        scale_uV=float(ns.scale_uv),
        linewidth=float(ns.linewidth),
        major_sec=float(ns.major_sec),
        minor_sec=float(ns.minor_sec),
        pred_alpha=float(ns.pred_alpha),
        ann_alpha=float(ns.ann_alpha),
        show_pred_bar=(not ns.no_pred_bar),
        pdf_dpi=int(ns.pdf_dpi),
        rasterize_traces=(not ns.no_raster),
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)


# if __name__ == "__main__":
#     # ---- EDIT HERE (minimal) ----
#     cfg = PlotCfg(
#         set_path="/Users/whatislove/study/phd/data/eegs_classification/fon/Co_y6_059_fon1_clean.set",
#         intervals_json="intervals.json",
#         out_pdf="eeg_with_preds.pdf",

#         # Pretty display defaults:
#         plot_sfreq=250.0,   # plotting only
#         l_freq=0.5,
#         h_freq=40.0,
#         notch=50.0,         # set 60 if действительно нужно

#         channel_mode="clinical_19",  # or posterior / frontal / first_32
#         page_sec=10.0,       # like clinical 10s/page
#         scale_uV=50.0,       # adjust if traces too tall/flat


#     )
#     main(cfg)




# ❯ python scripts/plot_preds.py \
#   --set '/Users/whatislove/study/phd/data/eegs_classification/fon/Co_y6_059_fon1_clean.set' \
#   --intervals intervals.json \
#   --out eeg_preds.pdf \
#   --plot-sfreq 200 \--panel-sec 30 --panels-per-page 4 \
#   --scale-uv 80 \
#   --linewidth 0.45 \
#   --major-sec 10 --minor-sec 2 \
#   --pred-alpha 0.06 \
#   --no-raster
# EEG channel type selected for re-referencing
# Applying average reference.
# Applying a custom ('EEG',) reference.
# Saved: eeg_preds.pdf
# ❯ python scripts/plot_preds.py \
#   --set '/Users/whatislove/study/phd/data/eegs_classification/fon/Co_y6_059_fon1_clean.set' \
#   --intervals intervals.json \
#     --out eeg_preds_small.pdf \
#   --plot-sfreq 125 \
#   --panel-sec 30 --panels-per-page 4 \
#   --scale-uv 80 \
#   --linewidth 0.45 \
#   --pdf-dpi 350
# EEG channel type selected for re-referencing
# Applying average reference.
# Applying a custom ('EEG',) reference.
# Saved: eeg_preds_small.pdf
# ~/study/phd/eeg_classification main !7 ?11                                                                                                                                                  9s  eeg_clf  system
# ❯ 
