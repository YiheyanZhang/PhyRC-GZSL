"""Generate evidence-driven SVG figures for the PhyRC paper."""

import json
from html import escape
from pathlib import Path


VARIANTS = (
    "full",
    "no_relational_prototype",
    "no_cross_fitting",
    "no_dual_evidence",
    "no_risk_constraint",
)

TRADEOFF_KEYS = ("Seen_AA", "Unseen_AA", "OA", "H", "seen_zero")


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def is_rcjd_feasible(candidate, fallback):
    return (
        candidate["seen_zero"] <= fallback["seen_zero"]
        and candidate["Seen_AA"] >= fallback["Seen_AA"] - 1.0
        and candidate["OA"] >= fallback["OA"] - 0.5
        and candidate["H"] > fallback["H"]
        and candidate["Unseen_AA"] > fallback["Unseen_AA"]
    )


def _tradeoff_unit(result):
    selection = result["selection"]
    fallback = {key: selection["p1_proxy"][key] for key in TRADEOFF_KEYS}
    selected = {key: selection["selected"][key] for key in TRADEOFF_KEYS}
    candidates = []
    for row in selection["rcjd_candidates"]:
        candidate = {key: row[key] for key in TRADEOFF_KEYS}
        candidate["feasible"] = is_rcjd_feasible(candidate, fallback)
        candidates.append(candidate)
    return {"fallback": fallback, "selected": selected, "candidates": candidates}


def load_loco_tradeoff_data(project_root):
    checkpoint_root = Path(project_root) / "checkpoints"
    data = {}
    for dataset in ("paviau", "houston"):
        documents = [_read(checkpoint_root / f"{dataset}_phyrc_single_unseen.json")]
        documents += [
            _read(checkpoint_root / "multiseed" / dataset / f"seed{seed}" / f"{dataset}_phyrc.json")
            for seed in range(43, 47)
        ]
        data[dataset] = [
            _tradeoff_unit(result)
            for document in documents
            for result in document["results"]
        ]
    data["indian_pines"] = [
        _tradeoff_unit(_read(path)["result"])
        for path in sorted(
            (checkpoint_root / "indian_pines_submission" / "single").glob("seed*/phyrc_s*.json")
        )
    ]
    return data


def write_loco_tradeoff_figure(project_root, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from statistics import fmean, pstdev

    data = load_loco_tradeoff_data(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "phyrc_loco_tradeoff.pdf"
    png_path = output_dir / "phyrc_loco_tradeoff.png"
    colors = {"infeasible": "#A7A9AC", "feasible": "#0072B2", "selected": "#D55E00"}

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.7,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharex=True, sharey=True)
        grid = np.linspace(0.1, 100.0, 300)
        seen_grid, unseen_grid = np.meshgrid(grid, grid)
        h_grid = 2 * seen_grid * unseen_grid / (seen_grid + unseen_grid)

        for index, (dataset, title) in enumerate((
            ("paviau", "(a) PaviaU"),
            ("houston", "(b) Houston"),
            ("indian_pines", "(c) Indian Pines"),
        )):
            axis = axes[index]
            units = data[dataset]
            candidates = [candidate for unit in units for candidate in unit["candidates"]]
            infeasible = [candidate for candidate in candidates if not candidate["feasible"]]
            feasible = [candidate for candidate in candidates if candidate["feasible"]]

            contours = axis.contour(
                seen_grid, unseen_grid, h_grid, levels=(20, 40, 60, 80),
                colors="#D7DADF", linewidths=0.55, linestyles="--", zorder=0,
            )
            axis.clabel(contours, fmt=lambda value: f"H={int(value)}", fontsize=5.4, inline=True)
            axis.scatter(
                [row["Seen_AA"] for row in infeasible],
                [row["Unseen_AA"] for row in infeasible],
                s=5, marker="x", linewidths=0.35, color=colors["infeasible"], alpha=0.16,
                zorder=1,
            )
            axis.scatter(
                [row["Seen_AA"] for row in feasible],
                [row["Unseen_AA"] for row in feasible],
                s=7, marker="o", linewidths=0.45, facecolors="none",
                edgecolors=colors["feasible"], alpha=0.30, zorder=2,
            )

            for key, marker, color, zorder in (
                ("fallback", "D", "#111111", 4),
                ("selected", "*", colors["selected"], 5),
            ):
                seen_values = [unit[key]["Seen_AA"] for unit in units]
                unseen_values = [unit[key]["Unseen_AA"] for unit in units]
                axis.errorbar(
                    fmean(seen_values), fmean(unseen_values),
                    xerr=pstdev(seen_values), yerr=pstdev(unseen_values),
                    fmt=marker, markersize=7.5 if marker == "*" else 5.2,
                    markerfacecolor=color, markeredgecolor="white" if marker == "*" else color,
                    markeredgewidth=0.55, ecolor=color, elinewidth=0.75, capsize=2,
                    zorder=zorder,
                )

            axis.set_title(title, pad=3)
            axis.set_xlim(0, 100)
            axis.set_ylim(0, 100)
            axis.set_xticks((0, 25, 50, 75, 100))
            axis.set_yticks((0, 25, 50, 75, 100))
            axis.grid(False)
            axis.set_aspect("equal", adjustable="box")
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_linewidth(0.7)

        handles = [
            Line2D([], [], color=colors["infeasible"], marker="x", linestyle="None", markersize=4,
                   label="Infeasible candidate"),
            Line2D([], [], color=colors["feasible"], marker="o", markerfacecolor="none",
                   linestyle="None", markersize=4, label="Feasible candidate"),
            Line2D([], [], color=colors["selected"], marker="*", linestyle="None", markersize=7,
                   label="Selected (mean $\\pm$ SD)"),
            Line2D([], [], color="#111111", marker="D", linestyle="None", markersize=4,
                   label="Fallback (mean $\\pm$ SD)"),
        ]
        figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                      bbox_to_anchor=(0.5, 0.995), handletextpad=0.35, columnspacing=1.0)
        figure.supxlabel("LOCO seen-class AA (%)", y=0.035)
        figure.supylabel("LOCO pseudo-unseen-class AA (%)", x=0.008)
        figure.subplots_adjust(left=0.072, right=0.995, bottom=0.20, top=0.82, wspace=0.12)
        figure.savefig(pdf_path, facecolor="white")
        figure.savefig(png_path, dpi=600, facecolor="white")
        plt.close(figure)
    return pdf_path, png_path


def load_risk_data(project_root):
    summary = _read(
        Path(project_root) / "checkpoints" / "ablations" / "phyrc" / "summary.json"
    )
    datasets = {}
    for dataset in ("paviau", "houston", "longkou"):
        datasets[dataset] = {}
        for variant in VARIANTS:
            row = summary[variant][dataset]
            datasets[dataset][variant] = {
                "seen_mean": row["Seen_AA"]["mean"],
                "seen_std": row["Seen_AA"]["std"],
                "unseen_mean": row["Unseen_AA"]["mean"],
                "unseen_std": row["Unseen_AA"]["std"],
                "seen_zero": row["seen_zero"],
            }
    return datasets


def load_failure_data(project_root):
    statistics = _read(
        Path(project_root)
        / "checkpoints"
        / "statistics"
        / "phyrc_paired_bootstrap.json"
    )
    return statistics["datasets"]


DISPLAY_NAMES = {
    "paviau": "PaviaU", "houston": "Houston", "longkou": "LongKou",
    "indian_pines": "Indian Pines",
}
VARIANT_LABELS = {
    "full": "Full",
    "no_relational_prototype": "No relational prototype",
    "no_cross_fitting": "No cross-fitting",
    "no_dual_evidence": "No dual evidence",
    "no_risk_constraint": "No risk constraint",
}
RISK_LABEL_OFFSETS = {
    "paviau": {
        "full": (9, -12, "start"),
        "no_relational_prototype": (9, 20, "start"),
        "no_cross_fitting": (9, 18, "start"),
        "no_dual_evidence": (9, 20, "start"),
        "no_risk_constraint": (9, -12, "start"),
    },
    "houston": {
        "full": (9, -12, "start"),
        "no_relational_prototype": (9, 22, "start"),
        "no_cross_fitting": (9, 19, "start"),
        "no_dual_evidence": (-9, 23, "end"),
        "no_risk_constraint": (9, -12, "start"),
    },
    "longkou": {
        "full": (9, -12, "start"),
        "no_relational_prototype": (9, 19, "start"),
        "no_cross_fitting": (9, 23, "start"),
        "no_dual_evidence": (9, 20, "start"),
        "no_risk_constraint": (-9, -12, "end"),
    },
}


def _svg_start(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}"
        ".title{font-size:24px;font-weight:700}.panel{font-size:18px;font-weight:700}"
        ".axis{font-size:14px}.small{font-size:12px}.label{font-size:13px;font-weight:600}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="31" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def _line(parts, x1, y1, x2, y2, stroke="#697386", width=1.5, dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dashed}/>'
    )


def _risk_svg(data):
    width, height = 1500, 540
    parts = _svg_start(width, height, "Seen--Unseen Risk Trade-off under Controlled Ablations")
    colors = {
        "full": "#0072B2",
        "no_risk_constraint": "#D55E00",
        "other": "#7A8496",
    }
    for panel, dataset in enumerate(("paviau", "houston", "longkou")):
        left, top, plot_w, plot_h = 85 + panel * 490, 75, 390, 340
        rows = data[dataset]
        x_values = [row["seen_mean"] + sign * row["seen_std"] for row in rows.values() for sign in (-1, 1)]
        y_values = [row["unseen_mean"] + sign * row["unseen_std"] for row in rows.values() for sign in (-1, 1)]
        x_min, x_max = min(x_values) - 2, max(x_values) + 2
        y_min, y_max = max(0, min(y_values) - 5), min(100, max(y_values) + 5)
        sx = lambda value: left + (value - x_min) * plot_w / (x_max - x_min)
        sy = lambda value: top + plot_h - (value - y_min) * plot_h / (y_max - y_min)

        parts.append(
            f'<text x="{left + plot_w / 2}" y="59" text-anchor="middle" class="panel">'
            f'{DISPLAY_NAMES[dataset]}</text>'
        )
        for tick in range(5):
            x_value = x_min + tick * (x_max - x_min) / 4
            y_value = y_min + tick * (y_max - y_min) / 4
            _line(parts, sx(x_value), top, sx(x_value), top + plot_h, "#E4E8EF", 1)
            _line(parts, left, sy(y_value), left + plot_w, sy(y_value), "#E4E8EF", 1)
            parts.append(
                f'<text x="{sx(x_value):.1f}" y="{top + plot_h + 21}" text-anchor="middle" class="small">'
                f'{x_value:.0f}</text>'
            )
            parts.append(
                f'<text x="{left - 10}" y="{sy(y_value) + 4:.1f}" text-anchor="end" class="small">'
                f'{y_value:.0f}</text>'
            )
        _line(parts, left, top + plot_h, left + plot_w, top + plot_h, "#172033", 2)
        _line(parts, left, top, left, top + plot_h, "#172033", 2)

        full, unconstrained = rows["full"], rows["no_risk_constraint"]
        _line(
            parts,
            sx(full["seen_mean"]), sy(full["unseen_mean"]),
            sx(unconstrained["seen_mean"]), sy(unconstrained["unseen_mean"]),
            "#D55E00", 2, "7,5",
        )
        for variant, row in rows.items():
            x, y = sx(row["seen_mean"]), sy(row["unseen_mean"])
            x_low, x_high = sx(row["seen_mean"] - row["seen_std"]), sx(row["seen_mean"] + row["seen_std"])
            y_low, y_high = sy(row["unseen_mean"] - row["unseen_std"]), sy(row["unseen_mean"] + row["unseen_std"])
            color = colors.get(variant, colors["other"])
            _line(parts, x_low, y, x_high, y, color, 1.4)
            _line(parts, x, y_low, x, y_high, color, 1.4)
            _line(parts, x_low, y - 4, x_low, y + 4, color, 1.4)
            _line(parts, x_high, y - 4, x_high, y + 4, color, 1.4)
            _line(parts, x - 4, y_low, x + 4, y_low, color, 1.4)
            _line(parts, x - 4, y_high, x + 4, y_high, color, 1.4)
            if variant == "full":
                parts.append(
                    f'<polygon points="{x:.1f},{y-9:.1f} {x+8:.1f},{y+7:.1f} {x-8:.1f},{y+7:.1f}" '
                    f'fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
                )
                label = "Full"
            elif variant == "no_risk_constraint":
                parts.append(
                    f'<polygon points="{x:.1f},{y-8:.1f} {x+8:.1f},{y:.1f} {x:.1f},{y+8:.1f} {x-8:.1f},{y:.1f}" '
                    f'fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
                )
                label = f'No risk constraint (Seen=0: {row["seen_zero"]})'
            else:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
                label = {
                    "no_relational_prototype": "No RP",
                    "no_cross_fitting": "No CF",
                    "no_dual_evidence": "No DE",
                }[variant]
            dx, dy, anchor = RISK_LABEL_OFFSETS[dataset][variant]
            parts.append(
                f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" class="small" '
                f'fill="{color}">{escape(label)}</text>'
            )

        parts.append(
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 48}" text-anchor="middle" class="axis">'
            "Seen-class AA (%)</text>"
        )
        parts.append(
            f'<text x="{left - 57}" y="{top + plot_h / 2}" text-anchor="middle" class="axis" '
            f'transform="rotate(-90 {left - 57} {top + plot_h / 2})">Unseen-class AA (%)</text>'
        )
    parts.append(
        '<text x="750" y="520" text-anchor="middle" class="small">'
        "Error bars: population standard deviation over seeds 42--46. Dashed segment isolates the risk-constraint trade-off.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _failure_svg(data):
    width, height = 1500, 600
    parts = _svg_start(width, height, "Per-Class Unseen Accuracy and Cross-Seed Stability")
    for panel, dataset in enumerate(("paviau", "houston", "indian_pines")):
        left, top, plot_w, plot_h = 85 + panel * 490, 75, 390, 375
        rows = data[dataset]["per_class"]
        hard_ids = {row["class_id"] for row in data[dataset]["hardest_three"]}
        sx = lambda index: left + (index + 0.5) * plot_w / len(rows)
        sy = lambda value: top + plot_h - (value + 5) * plot_h / 110
        parts.append(
            f'<text x="{left + plot_w / 2}" y="59" text-anchor="middle" class="panel">'
            f'{DISPLAY_NAMES[dataset]}</text>'
        )
        for y_value in (0, 25, 50, 75, 100):
            _line(parts, left, sy(y_value), left + plot_w, sy(y_value), "#E4E8EF", 1)
            parts.append(
                f'<text x="{left - 10}" y="{sy(y_value) + 4:.1f}" text-anchor="end" class="small">'
                f'{y_value}</text>'
            )
        _line(parts, left, top + plot_h, left + plot_w, top + plot_h, "#172033", 2)
        _line(parts, left, top, left, top + plot_h, "#172033", 2)
        for index, row in enumerate(rows):
            x = sx(index)
            mean, std = row["mean_unseen_aa"], row["std_unseen_aa"]
            y, y_low, y_high = sy(mean), sy(mean - std), sy(mean + std)
            hard = row["class_id"] in hard_ids
            color = "#D55E00" if hard else "#0072B2"
            _line(parts, x, y_low, x, y_high, color, 1.4)
            _line(parts, x - 4, y_low, x + 4, y_low, color, 1.4)
            _line(parts, x - 4, y_high, x + 4, y_high, color, 1.4)
            if hard:
                parts.append(
                    f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="{color}"/>'
                )
                hard_rank = list(sorted(hard_ids)).index(row["class_id"])
                label_y = max(top + 14, y - 14 - 14 * hard_rank)
                parts.append(
                    f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" class="small" fill="{color}">'
                    f'{escape(row["class_name"])}</text>'
                )
            else:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" class="small">'
                f'{row["class_id"]}</text>'
            )
        parts.append(
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 45}" text-anchor="middle" class="axis">'
            "Held-out class ID</text>"
        )
        parts.append(
            f'<text x="{left - 57}" y="{top + plot_h / 2}" text-anchor="middle" class="axis" '
            f'transform="rotate(-90 {left - 57} {top + plot_h / 2})">Unseen accuracy (%)</text>'
        )
    parts.append(
        '<circle cx="550" cy="555" r="5" fill="#0072B2"/><text x="565" y="560" class="axis">All classes</text>'
        '<rect x="730" y="550" width="10" height="10" fill="#D55E00"/>'
        '<text x="747" y="560" class="axis">Three lowest five-seed means</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_figures(project_root, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_path = output_dir / "phyrc_risk_tradeoff.svg"
    failure_path = output_dir / "phyrc_failure_stability.svg"
    risk_path.write_text(_risk_svg(load_risk_data(project_root)), encoding="utf-8")
    failure_path.write_text(_failure_svg(load_failure_data(project_root)), encoding="utf-8")
    return risk_path, failure_path


def main():
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "AAAI_Press_LaTeX_Template" / "figures"
    for path in write_figures(project_root, output_dir):
        print(path)


if __name__ == "__main__":
    main()
