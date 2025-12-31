from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mne


@dataclass
class PlotConfig:
    set_path: str
    out_png: str = "eeg_paperstyle.png"
    out_pdf: str = "eeg_paperstyle.pdf"

    # What EEG window to show (seconds)
    t_start: float = 0.0
    duration: float = 120.0

    # Preprocessing for nicer-looking “paper” plots
    l_freq: Optional[float] = 0.5
    h_freq: Optional[float] = 40.0
    notch: Optional[float] = 60.0  # set to 60 if your mains is 60 Hz, or None

    # Montage guess (used if channel locations are missing/partial)
    montage_name: str = "standard_1020"  # try "standard_1005" if needed

    # How many channels to show in the stacked trace (None = all)
    max_traces: int = 32

    # Trace scaling robustness
    scale_quantile: float = 0.95

    # Event appearance
    event_line_alpha: float = 0.75
    event_line_width: float = 1.2


def _safe_str(x) -> str:
    return "" if x is None else str(x)


def load_set_as_raw(set_path: str) -> mne.io.BaseRaw:
    # EEGLAB loader
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")

    # If EEGLAB has annotations, they usually appear here:
    # raw.annotations: onset (sec), duration, description (string)
    return raw


def ensure_montage(raw: mne.io.BaseRaw, montage_name: str) -> None:
    # If the file already contains chanlocs, MNE may have positions.
    # Setting a standard montage helps layouts/topomaps when positions are missing.
    raw.rename_channels(lambda s: s.strip())  # убрать пробелы
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="warn")


def preprocess_for_plot(raw: mne.io.BaseRaw, l_freq: Optional[float], h_freq: Optional[float],
                        notch: Optional[float]) -> mne.io.BaseRaw:
    r = raw.copy()

    # Pick EEG-like channels if present (keeps EOG/ECG out unless you want them)
    # If your data uses nonstandard types, you can remove this or adjust.
    picks = mne.pick_types(r.info, eeg=True, eog=False, ecg=False, emg=False, misc=True)
    r.pick(picks)

    # Basic referencing improves readability (doesn't change your saved original)
    try:
        r.set_eeg_reference("average", projection=False)
    except Exception:
        pass

    # Filtering for “paper” readability
    if l_freq is not None or h_freq is not None:
        r.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")

    if notch is not None:
        try:
            r.notch_filter(freqs=[notch, 2 * notch], verbose="ERROR")
        except Exception:
            r.notch_filter(freqs=[notch], verbose="ERROR")

    return r


def annotations_to_events(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str, int]]:
    # Convert annotation strings → integer event codes
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    return events, event_id


def pick_traces_for_plot(raw: mne.io.BaseRaw, max_traces: int) -> Sequence[int]:
    n = raw.info["nchan"]
    if max_traces is None or max_traces >= n:
        return list(range(n))
    # Prefer EEG channels first if types are set correctly
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)
    if len(eeg_picks) == 0:
        return list(range(min(max_traces, n)))
    return list(eeg_picks[:max_traces])


def plot_stacked_traces(ax, raw: mne.io.BaseRaw, t0: float, duration: float,
                        trace_picks: Sequence[int], events: Optional[np.ndarray],
                        event_id: Optional[Dict[str, int]], cfg: PlotConfig) -> None:
    sfreq = float(raw.info["sfreq"])
    t1 = t0 + duration
    start_samp = int(round(t0 * sfreq))
    stop_samp = int(round(t1 * sfreq))

    data = raw.get_data(picks=trace_picks, start=start_samp, stop=stop_samp)
    times = np.arange(data.shape[1]) / sfreq + t0

    # Robust scale so channels have comparable visual amplitude
    # (using quantile of absolute values across all chosen channels)
    q = np.quantile(np.abs(data), cfg.scale_quantile)
    if q <= 0:
        q = 1.0
    data = data / q

    # Vertical offsets
    n_tr = data.shape[0]
    offsets = np.arange(n_tr)[::-1]  # top to bottom
    for i in range(n_tr):
        ax.plot(times, data[i] + offsets[i], linewidth=0.8)

    # Channel labels on y-axis
    ch_names = [raw.ch_names[p] for p in trace_picks]
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names, fontsize=8)

    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)")
    ax.set_title("EEG segment (stacked, normalized) with event markers", fontsize=11)

    # Event markers within window
    if events is not None and event_id is not None and len(events) > 0:
        inv_event_id = {v: k for k, v in event_id.items()}

        # events[:, 0] are sample indices (in raw time base)
        ev_t = events[:, 0] / sfreq
        in_win = (ev_t >= t0) & (ev_t <= t1)
        ev_t = ev_t[in_win]
        ev_code = events[in_win, 2] if np.any(in_win) else np.array([])

        # Color by event code (simple deterministic palette)
        uniq = np.unique(ev_code) if ev_code.size else []
        cmap = plt.get_cmap("tab10")
        code2color = {c: cmap(i % 10) for i, c in enumerate(uniq)}

        for t, c in zip(ev_t, ev_code):
            ax.axvline(t, color=code2color.get(c, "k"),
                       alpha=cfg.event_line_alpha, linewidth=cfg.event_line_width)

        # Legend (small)
        if len(uniq) > 0:
            handles = []
            labels = []
            for c in uniq[:10]:
                name = inv_event_id.get(int(c), str(int(c)))
                h = plt.Line2D([0], [0], color=code2color[c], linewidth=2)
                handles.append(h)
                labels.append(name)
            ax.legend(handles, labels, loc="upper right", fontsize=7, frameon=True)


def plot_sensor_topomap(ax, raw: mne.io.BaseRaw) -> None:
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    if len(picks) == 0:
        ax.text(0.5, 0.5, "No EEG picks for topomap", ha="center", va="center")
        ax.axis("off")
        return

    # Работаем с под-raw, чтобы порядок каналов совпадал с metric
    raw_sub = raw.copy().pick(picks)

    data = raw_sub.get_data()          # (n_ch, n_times)
    metric = np.std(data, axis=1)      # любая метрика на канал, здесь STD

    try:
        # В одних версиях MNE можно передавать Info напрямую
        mne.viz.plot_topomap(metric, raw_sub.info, axes=ax, show=False, contours=0)
    except TypeError:
        # В других версиях нужно передавать 2D позиции (pos), а не Info
        layout = mne.channels.find_layout(raw_sub.info)
        pos2d = layout.pos[:, :2]
        mne.viz.plot_topomap(metric, pos2d, axes=ax, show=False, contours=0)

    ax.set_title("Sensor map (metric: channel STD)", fontsize=11)


def plot_psd_panel(ax, raw: mne.io.BaseRaw) -> None:
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    if len(picks) == 0:
        ax.text(0.5, 0.5, "No EEG picks for PSD", ha="center", va="center")
        ax.axis("off")
        return

    data = raw.get_data(picks=picks)          # shape (n_ch, n_times)
    sfreq = float(raw.info["sfreq"])

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=0.5,
        fmax=45.0,
        n_fft=2048,
        n_overlap=1024,   # optional; set 0 if you prefer no overlap
        average="mean",
    )
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
    mean_db = psd_db.mean(axis=0)

    ax.plot(freqs, mean_db, linewidth=1.5)
    ax.set_title("Mean PSD (Welch, dB)", fontsize=11)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.25)


def plot_metadata_block(ax, raw: mne.io.BaseRaw, events: np.ndarray, event_id: Dict[str, int],
                        cfg: PlotConfig) -> None:
    info = raw.info
    sfreq = info.get("sfreq", None)
    nchan = info.get("nchan", None)

    # Some fields may be None depending on file
    subj = getattr(info, "subject_info", None)
    subj_str = ""
    if isinstance(subj, dict):
        parts = []
        for k in ["his_id", "id", "sex", "hand", "birthday"]:
            if k in subj and subj[k] is not None:
                parts.append(f"{k}={subj[k]}")
        subj_str = ", ".join(parts)

    meas_date = info.get("meas_date", None)

    # Annotation summary
    n_annot = len(raw.annotations) if raw.annotations is not None else 0
    n_events = int(events.shape[0]) if events is not None else 0
    event_names = list(event_id.keys())[:12] if event_id else []

    lines = [
        "Metadata",
        "—" * 32,
        f"File: {os.path.basename(cfg.set_path)}",
        f"Sampling rate: {_safe_str(sfreq)} Hz",
        f"Channels: {_safe_str(nchan)}",
        f"meas_date: {_safe_str(meas_date)}",
        f"Subject: {subj_str if subj_str else 'n/a'}",
        "",
        "Events/Annotations",
        "—" * 32,
        f"Annotations: {n_annot}",
        f"Events: {n_events}",
        f"Event types (first): {', '.join(event_names) if event_names else 'n/a'}",
        "",
        "Plot settings",
        "—" * 32,
        f"Window: {cfg.t_start:.2f}s → {cfg.t_start + cfg.duration:.2f}s",
        f"Bandpass: {cfg.l_freq}–{cfg.h_freq} Hz",
        f"Notch: {cfg.notch} Hz",
        f"Montage (fallback): {cfg.montage_name}",
    ]

    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    ax.axis("off")


def make_paper_style_figure(cfg: PlotConfig) -> None:
    raw = load_set_as_raw(cfg.set_path)

    print(raw.get_montage())
    print("dig:", raw.info["dig"] is not None, len(raw.info["dig"] or []))
    pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    print("channels with pos:", (np.linalg.norm(pos, axis=1) > 0).sum())

    ensure_montage(raw, cfg.montage_name)

    raw_plot = preprocess_for_plot(raw, cfg.l_freq, cfg.h_freq, cfg.notch)
    events, event_id = annotations_to_events(raw_plot)

    trace_picks = pick_traces_for_plot(raw_plot, cfg.max_traces)

    # Figure layout (mosaic)
    mosaic = [
        ["traces",  "traces",  "meta"],
        ["topo",    "psd",     "meta"],
    ]
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=(14, 8), constrained_layout=True
    )

    plot_stacked_traces(axd["traces"], raw_plot, cfg.t_start, cfg.duration,
                        trace_picks, events, event_id, cfg)
    plot_sensor_topomap(axd["topo"], raw_plot)
    plot_psd_panel(axd["psd"], raw_plot)
    plot_metadata_block(axd["meta"], raw_plot, events, event_id, cfg)

    fig.suptitle("EEG paper-style summary", fontsize=14)

    fig.savefig(cfg.out_png, dpi=300, bbox_inches="tight")
    fig.savefig(cfg.out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved:\n- {cfg.out_png}\n- {cfg.out_pdf}")


if __name__ == "__main__":
    # EDIT THIS
    cfg = PlotConfig(
        set_path='/Users/whatislove/study/phd/data/eegs_classification/fon/Co_y6_059_fon1_clean.set',
        t_start=0.0,
        duration=500.0,
        notch=50.0,            # Sweden/EU is typically 50 Hz mains
        l_freq=0.5,
        h_freq=40.0,
        max_traces=32,
        montage_name="standard_1005",
        out_png="eeg_paperstyle.png",
        out_pdf="eeg_paperstyle.pdf",
    )
    make_paper_style_figure(cfg)
