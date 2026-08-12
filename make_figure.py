"""
Figure 1: per-tag macro-F1 across the four feature conditions.

Reads output/results.json (the averaged_* entries, which carry per-tag F1),
draws one grouped bar per POS tag with the four conditions ordered by
increasing information (morphology-only -> full), and writes a vector PDF
for LaTeX plus a PNG for quick viewing.

The visual story: tags are ordered by descending support, so high-frequency
tags (flat, near-ceiling) sit on the left and rare tags (which collapse as
features are stripped) sit on the right.

Run from the repo root:
    python3 make_figure.py
Requires matplotlib (pip install matplotlib --break-system-packages if missing).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, 'output', 'results.json')
OUT_PDF = os.path.join(REPO, 'output', 'fig_pertag_f1.pdf')
OUT_PNG = os.path.join(REPO, 'output', 'fig_pertag_f1.png')

# (results.json key, legend label), ordered by increasing information
CONDITIONS = [
    ('averaged_morph',   'Morphology only'),
    ('averaged_nolex',   '+ neighbour morphology'),
    ('averaged_delex',   '+ neighbour words'),
    ('averaged_lexical', '+ target word (full)'),
]
# sequential shades: light = least information, dark = most
COLORS = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
EXCLUDE = {'X'}


def main():
    if not os.path.exists(RESULTS):
        sys.exit(f"FAIL: {RESULTS} not found. Run run_all.sh first, from the repo root.")

    with open(RESULTS) as fh:
        results = json.load(fh)

    for key, _ in CONDITIONS:
        if key not in results or 'per_tag' not in results[key]:
            sys.exit(f"FAIL: results.json missing per-tag data for '{key}'. "
                     f"Re-run run_experiments.py.")

    # tag order: descending support, taken from the full model (support is
    # gold-derived and identical across conditions)
    ref = results['averaged_lexical']['per_tag']
    tags = [t for t in ref if t not in EXCLUDE]
    tags.sort(key=lambda t: -ref[t]['support'])

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        sys.exit("FAIL: matplotlib not installed. "
                 "pip install matplotlib --break-system-packages")

    x = np.arange(len(tags))
    n = len(CONDITIONS)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for i, (key, label) in enumerate(CONDITIONS):
        pt = results[key]['per_tag']
        vals = [pt.get(t, {}).get('f1_mean', 0.0) for t in tags]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
               label=label, color=COLORS[i], edgecolor='white', linewidth=0.3)

    ax.set_ylabel('Per-tag F1', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=8, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=7.5, ncol=2, loc='lower left', framealpha=0.9)
    ax.grid(axis='y', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print("\nUpload fig_pertag_f1.pdf to Overleaf and include with:")
    print(r"  \includegraphics[width=\linewidth]{fig_pertag_f1.pdf}")


if __name__ == '__main__':
    main()
