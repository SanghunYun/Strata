#!/usr/bin/env python3
"""Chart for the PR: IQ2_XS (results/v0.2.1) and IQ3_XXS (results/v0.2.1-iq3), ShareGPT 120, 3 runs per setting.

One row per model, three panels each over the 120 requests (as plot_v021.py): decode tok/s, prompt tok/s and expert
reads from the file per generated token, each point a 10-request window of the 3 runs per setting pooled together
(window sums). A column shares its y axis, so the two models compare directly.
  python3 charts/plot_v021_models.py   (matplotlib) -> charts/sharegpt120-v021-iq2-iq3.png
"""
import glob, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RESULTS = HERE.parent / 'results'
MODELS = [('IQ2_XS', 'v0.2.1', '3769 VRAM expert slots'), ('IQ3_XXS', 'v0.2.1-iq3', '2791 VRAM expert slots')]
# fixed categorical order (blue, orange, aqua, yellow); identity follows the setting, never its rank
SETTINGS = [('pr', 'this PR (RAM_ADAPT=1)', '#2a78d6'), ('ram-adapt-8', 'main RAM_ADAPT=8', '#eb6834'),
            ('ram-adapt2', 'main RAM_ADAPT2', '#1baf7a'), ('off', 'off', '#eda100')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
W = 10
T = lambda r: r['timings']
PANELS = [('Decode tok/s (higher is better)', lambda r: T(r)['predicted_n'], lambda r: T(r)['predicted_ms'] / 1000),
          ('Prompt tok/s (higher is better)', lambda r: T(r)['prompt_n'], lambda r: T(r)['prompt_ms'] / 1000),
          ('Expert reads from the file per token (lower is better)', lambda r: r['file_blobs'], lambda r: r['completion_tokens'])]

def load(tag):
    runs = {k: [[json.loads(l) for l in open(f)] for f in sorted(glob.glob(str(RESULTS / tag / f'sharegpt-{k}-[0-9].jsonl')))]
            for k, _, _ in SETTINGS}
    assert all(len(v) == 3 and all(len(r) == 120 for r in v) for v in runs.values()), tag
    return runs

def window(runs, num, den):
    out = []
    for i in range(120):
        lo = max(0, i - W + 1)
        n = sum(num(r) for rr in runs for r in rr[lo:i + 1]); d = sum(den(r) for rr in runs for r in rr[lo:i + 1])
        out.append(n / d if d else float('nan'))
    return out

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'axes.titlecolor': INK})
fig, axs = plt.subplots(2, 3, figsize=(16, 8.4), sharey='col', sharex=True)
fig.patch.set_facecolor(SURFACE)
x = list(range(1, 121))
for row, (model, tag, note) in enumerate(MODELS):
    runs = load(tag)
    for col, (title, num, den) in enumerate(PANELS):
        ax = axs[row][col]
        ax.set_facecolor(SURFACE); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for k, lab, c in SETTINGS:
            ax.plot(x, window(runs[k], num, den), color=c, lw=2, label=lab, solid_capstyle='round')
        ax.set_title(f'{model} - {title}', loc='left', fontsize=11)
        ax.tick_params(labelleft=True)
        if row == 1: ax.set_xlabel('request')
    axs[row][0].set_ylabel(f'{model} ({note})')
    for k, lab, _ in SETTINGS:
        tot = lambda num, den: sum(num(r) for rr in runs[k] for r in rr) / sum(den(r) for rr in runs[k] for r in rr)
        print(f'{model} {lab}: decode {tot(PANELS[0][1], PANELS[0][2]):.2f}, prompt {tot(PANELS[1][1], PANELS[1][2]):.1f}, '
              f'reads/token {tot(PANELS[2][1], PANELS[2][2]):.2f}')
for col in range(3):   # one y range per column, the larger of the two models
    top = max(max(l.get_ydata()) for row in axs for l in row[col].get_lines())
    axs[0][col].set_ylim(0, top * 1.05)
handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01, 'RTX 3060 12 GB, 32 GB RAM, NVMe, RAM tier 20.5 GiB, v0.2.1, ShareGPT 120 requests; the same engine '
         'arguments for both models except --pack/--native. Each point: a 10-request window over the 3 fresh-engine runs '
         'per setting (window sums).', fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(HERE / 'sharegpt120-v021-iq2-iq3.png', dpi=150, facecolor=SURFACE)
