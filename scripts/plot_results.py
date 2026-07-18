#!/usr/bin/env python3
"""Regenerate paper/figures/study_results_main.{png,pdf} from judgments CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "outputs" / "judgments_20260717.csv"
OUT_DIRS = [REPO / "paper" / "figures", REPO / "outputs" / "figures"]

MODELS = ["qwen7b", "llama", "phi4mini"]
MODEL_LABELS = {
    "qwen7b": "Qwen2.5-7B",
    "llama": "Llama-3.1-8B",
    "phi4mini": "Phi-4 Mini",
}
FRAMINGS = ["neutral", "catholic"]
ALIGN_ORDER = ["Aligned", "Partially aligned", "Misaligned", "Refusal"]
ALIGN_COLORS = {
    "Aligned": "#1B6B5A",
    "Partially aligned": "#C47E2A",
    "Misaligned": "#A33B3B",
    "Refusal": "#6B7280",
}
FRAME_COLORS = {"neutral": "#2C4A6E", "catholic": "#1B6B5A"}


def load_rows(path: Path):
    rows = [
        r
        for r in csv.DictReader(path.open(encoding="utf-8"))
        if (r.get("alignment") or "").strip() and not (r.get("error") or "").strip()
    ]
    if len(rows) != 300:
        raise SystemExit(f"Expected 300 successful judgments, found {len(rows)} in {path}")
    return rows


def subset(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def rate(rows, pred):
    return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else 0.0


def main() -> None:
    rows = load_rows(CSV)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F5F1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Dedicated rows so titles / legends / labels / axes never share a band.
    # [fig title][A title][A legend][A plot][A frame labels][BC titles][BC legend][B|C plots]
    fig = plt.figure(figsize=(11.4, 10.2))
    gs = fig.add_gridspec(
        9,
        2,
        height_ratios=[0.22, 0.18, 0.18, 1.35, 0.28, 0.2, 0.18, 1.15, 0.45],
        hspace=0.14,
        wspace=0.28,
        top=0.97,
        bottom=0.03,
        left=0.09,
        right=0.98,
    )

    # --- Figure title ---
    ax_ft = fig.add_subplot(gs[0, :])
    ax_ft.axis("off")
    ax_ft.text(
        0.5,
        0.5,
        "Theological anthropology in three open-weight LLMs (N = 300)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        transform=ax_ft.transAxes,
    )

    # --- Panel A title ---
    ax_at = fig.add_subplot(gs[1, :])
    ax_at.axis("off")
    ax_at.text(
        0.0,
        0.4,
        "A. Alignment (LLM-as-judge: gpt-5-mini), by model and framing",
        ha="left",
        va="center",
        fontsize=12,
        transform=ax_at.transAxes,
    )

    # --- Panel A legend (own row) ---
    ax_al = fig.add_subplot(gs[2, :])
    ax_al.axis("off")
    align_patches = [
        Patch(facecolor=ALIGN_COLORS[a], edgecolor="white", label=a) for a in ALIGN_ORDER
    ]
    ax_al.legend(
        handles=align_patches,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.4,
    )

    # --- Panel A plot ---
    ax = fig.add_subplot(gs[3, :])
    x = np.arange(len(MODELS))
    width = 0.36
    for i, fr in enumerate(FRAMINGS):
        bottoms = np.zeros(len(MODELS))
        offsets = x - width / 2 + i * width
        for align in ALIGN_ORDER:
            heights = np.array(
                [
                    100
                    * rate(
                        subset(rows, model_key=mk, framing=fr),
                        lambda r, a=align: r["alignment"] == a,
                    )
                    for mk in MODELS
                ]
            )
            ax.bar(
                offsets,
                heights,
                width=width * 0.92,
                bottom=bottoms,
                color=ALIGN_COLORS[align],
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms += heights
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)
    ax.set_ylabel("Share of responses (%)", fontsize=10)
    ax.set_ylim(0, 100)

    # --- Panel A framing labels (own row under A; keeps them off B/C titles) ---
    ax_af = fig.add_subplot(gs[4, :], sharex=ax)
    ax_af.axis("off")
    ax_af.set_xlim(ax.get_xlim())
    for i in range(len(MODELS)):
        ax_af.text(
            i - width / 2,
            0.55,
            "neutral",
            ha="center",
            va="center",
            fontsize=8,
            color="#2C4A6E",
            transform=ax_af.get_xaxis_transform(),
        )
        ax_af.text(
            i + width / 2,
            0.55,
            "Catholic",
            ha="center",
            va="center",
            fontsize=8,
            color="#1B6B5A",
            transform=ax_af.get_xaxis_transform(),
        )

    # --- Panels B/C titles ---
    ax_bt = fig.add_subplot(gs[5, 0])
    ax_bt.axis("off")
    ax_bt.text(
        0.0,
        0.4,
        "B. Secondary flags (pooled)",
        ha="left",
        va="center",
        fontsize=11,
        transform=ax_bt.transAxes,
    )
    ax_ct = fig.add_subplot(gs[5, 1])
    ax_ct.axis("off")
    ax_ct.text(
        0.0,
        0.4,
        "C. Item Q01 (ground of dignity): % Aligned",
        ha="left",
        va="center",
        fontsize=11,
        transform=ax_ct.transAxes,
    )

    # --- Shared framing legend for B and C ---
    ax_fl = fig.add_subplot(gs[6, :])
    ax_fl.axis("off")
    frame_patches = [
        Patch(facecolor=FRAME_COLORS["neutral"], edgecolor="white", label="Neutral"),
        Patch(facecolor=FRAME_COLORS["catholic"], edgecolor="white", label="Catholic"),
    ]
    ax_fl.legend(
        handles=frame_patches,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.6,
    )

    # --- Panel B plot ---
    ax2 = fig.add_subplot(gs[7, 0])
    flags = [
        ("invokes_religion", "Invokes\nreligion"),
        ("capacity_language", "Capacity\nlanguage"),
        ("relational_language", "Relational\nlanguage"),
        ("ranks_persons", "Ranks\npersons"),
    ]
    y = np.arange(len(flags))
    h = 0.36
    for i, fr in enumerate(FRAMINGS):
        vals = [
            100 * rate(subset(rows, framing=fr), lambda r, f=flag: r.get(f) == "true")
            for flag, _ in flags
        ]
        ax2.barh(
            y + (i - 0.5) * h,
            vals,
            height=h * 0.9,
            color=FRAME_COLORS[fr],
            edgecolor="white",
            linewidth=0.5,
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels([lab for _, lab in flags], fontsize=9)
    ax2.set_xlabel("Share of responses (%)", fontsize=9, labelpad=6)
    ax2.set_xlim(0, 100)

    # --- Panel C plot ---
    ax3 = fig.add_subplot(gs[7, 1])
    w = 0.35
    for i, fr in enumerate(FRAMINGS):
        vals = [
            100
            * rate(
                subset(rows, model_key=mk, framing=fr, item_id="Q01"),
                lambda r: r["alignment"] == "Aligned",
            )
            for mk in MODELS
        ]
        ax3.bar(
            x + (i - 0.5) * w,
            vals,
            width=w * 0.9,
            color=FRAME_COLORS[fr],
            edgecolor="white",
        )
        for j, v in enumerate(vals):
            if v < 0.5:
                ax3.plot(
                    [x[j] + (i - 0.5) * w],
                    [0],
                    marker="_",
                    markersize=14,
                    color=FRAME_COLORS[fr],
                    markeredgewidth=2,
                )
    ax3.set_xticks(x)
    ax3.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    ax3.set_ylabel("% Aligned", fontsize=9)
    ax3.set_ylim(0, 105)

    # --- Footer (own row; clears B/C x-axis labels) ---
    ax_ftnote = fig.add_subplot(gs[8, :])
    ax_ftnote.axis("off")
    ax_ftnote.text(
        0.5,
        0.15,
        "LLM-as-judge first pass (gpt-5-mini); awaits human adjudication. "
        "Neutral vs Catholic matched prompts × 5 runs.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
        transform=ax_ftnote.transAxes,
    )

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        base = d / "study_results_main"
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
        fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.15)
        print(f"wrote {base}.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
