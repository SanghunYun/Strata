#!/usr/bin/env python3
"""Chart for the PR from results/v0.2.1 (ShareGPT 120, 3 runs per setting).

Three panels over the 120 requests, each point a 10-request window of the 3 runs per setting pooled together:
decode tok/s (generated tokens / decode time, engine timings), prompt tok/s (prompt tokens read / prompt time) and
expert reads from the file per generated token. Window sums, not averages of per-request rates.
  python3 charts/plot_v021.py   (matplotlib) -> charts/sharegpt120-v021.png
"""
import glob, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RES = HERE.parent / 'results' / 'v0.2.1'
# fixed categorical order (blue, orange, aqua, yellow); identity follows the setting, never its rank
SETTINGS = [('pr', 'this PR (RAM_ADAPT=1)', '#2a78d6'), ('ram-adapt-8', 'main RAM_ADAPT=8', '#eb6834'),
            ('ram-adapt2', 'main RAM_ADAPT2', '#1baf7a'), ('off', 'off', '#eda100')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
W = 10

runs = {k: [[json.loads(l) for l in open(f)] for f in sorted(glob.glob(str(RES / f'sharegpt-{k}-[0-9].jsonl')))] for k, _, _ in SETTINGS}
assert all(len(v) == 3 and all(len(r) == 120 for r in v) for v in runs.values())

def window(k, num, den):
    out = []
    for i in range(120):
        lo = max(0, i - W + 1)
        n = sum(num(r) for rr in runs[k] for r in rr[lo:i + 1]); d = sum(den(r) for rr in runs[k] for r in rr[lo:i + 1])
        out.append(n / d if d else float('nan'))
    return out
T = lambda r: r['timings']
PANELS = [('Decode tok/s (higher is better)', lambda r: T(r)['predicted_n'], lambda r: T(r)['predicted_ms'] / 1000),
          ('Prompt tok/s (higher is better)', lambda r: T(r)['prompt_n'], lambda r: T(r)['prompt_ms'] / 1000),
          ('Expert reads from the file per token (lower is better)', lambda r: r['file_blobs'], lambda r: r['completion_tokens'])]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'axes.titlecolor': INK})
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
fig.patch.set_facecolor(SURFACE)
x = list(range(1, 121))
for ax, (title, num, den) in zip(axs, PANELS):
    ax.set_facecolor(SURFACE); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for k, lab, c in SETTINGS:
        ax.plot(x, window(k, num, den), color=c, lw=2, label=lab, solid_capstyle='round')
    ax.set_title(title, loc='left', fontsize=11); ax.set_xlabel('request'); ax.set_ylim(0, None)
axs[2].legend(frameon=False, loc='upper left')
fig.text(0.01, 0.01, 'RTX 3060 12 GB, 32 GB RAM, NVMe, IQ2_XS, RAM tier 20.5 GiB, v0.2.1, ShareGPT 120 requests. '
         'Each point: a 10-request window over the 3 fresh-engine runs per setting (window sums).', fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(HERE / 'sharegpt120-v021.png', dpi=150, facecolor=SURFACE)
for k, lab, _ in SETTINGS:
    tot = lambda num, den: sum(num(r) for rr in runs[k] for r in rr) / sum(den(r) for rr in runs[k] for r in rr)
    print(f'{lab}: decode {tot(PANELS[0][1], PANELS[0][2]):.2f}, prompt {tot(PANELS[1][1], PANELS[1][2]):.1f}, reads/token {tot(PANELS[2][1], PANELS[2][2]):.2f}')
