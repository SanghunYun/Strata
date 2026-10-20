#!/usr/bin/env python3
"""Compare settings from repeated replays of the same requests.

  python3 summarize.py --ref a8 off='runs/sg-off-*.jsonl' a8='runs/sg-a8-*.jsonl' a2='runs/sg-a2-*.jsonl' pr='runs/sg-pr-*.jsonl'

Each NAME=GLOB is one setting; each matching file is one run (a fresh engine). Speeds are the engine's own timings
from the last stream chunk (decode = predicted_n / predicted_ms, prompt = prompt_n / prompt_ms, summed over the run);
without them the client-side estimate completion tokens / (total - TTFT) is used and the line says so.
Per setting: mean of the runs and their A/A range. Every setting against --ref: change of the means with a Welch 95%
interval. Storage read GB comes from <run>.meta.json (replay.py --engine-pid). A 120-request run is also split into
request blocks 1-10 / 11-30 / 31-60 / 61-120. Inputs must be identical across all runs (body hashes are checked).
"""
import argparse, glob, json, math, statistics as st, sys

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
BLOCKS = [(0, 10), (10, 30), (30, 60), (60, 120)]

def load(p): return [json.loads(x) for x in open(p)]
def rates(rs):
    t = [r.get('timings') or {} for r in rs]
    if all(x.get('predicted_ms') for x in t):
        dec = sum(x.get('predicted_n') or 0 for x in t) / (sum(x['predicted_ms'] for x in t) / 1000)
        pm = sum(x.get('prompt_ms') or 0 for x in t)
        pre = sum(x.get('prompt_n') or 0 for x in t) / (pm / 1000) if pm else float('nan')
        return dec, pre, 'engine'
    tok = sum(r['completion_tokens'] for r in rs); return tok / sum(r['total_s'] - r['ttft_s'] for r in rs), float('nan'), 'client'
def run_stats(p):
    rs = load(p); dec, pre, src = rates(rs)
    s = {'file': p, 'hashes': [r['body_sha256'] for r in rs], 'decode_tps': dec, 'prompt_tps': pre, 'source': src,
         'service_s': sum(r['total_s'] for r in rs), 'ttft_p50': st.median(r['ttft_s'] for r in rs)}
    try:
        m = json.load(open(p + '.meta.json'))
        s['read_gb'] = (m['io_end']['read_bytes'] - m['io_start']['read_bytes']) / 1e9 if m.get('io_start') else None
    except OSError: s['read_gb'] = None
    if len(rs) == 120: s['blocks'] = [rates(rs[a:b])[0] for a, b in BLOCKS]
    return s
def welch(a, b):
    ma, mb = st.mean(a), st.mean(b)
    if len(a) < 2 or len(b) < 2: return 100 * (mb / ma - 1), None, None
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b); se = math.sqrt(va + vb)
    df = se ** 4 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se else 99
    t = T975.get(min(10, max(1, math.floor(df))), 2.228) if df < 30 else 1.96
    return 100 * (mb - ma) / ma, 100 * (mb - ma - t * se) / ma, 100 * (mb - ma + t * se) / ma

ap = argparse.ArgumentParser(); ap.add_argument('--ref'); ap.add_argument('settings', nargs='+')
a = ap.parse_args()
S = {}
for spec in a.settings:
    name, pat = spec.split('=', 1); files = sorted(glob.glob(pat)); assert files, spec
    S[name] = [run_stats(f) for f in files]
first = next(iter(S.values()))[0]['hashes']
assert all(r['hashes'] == first for v in S.values() for r in v), 'the runs did not send identical requests'
ref = a.ref or next(iter(S))
METRICS = ['decode_tps', 'prompt_tps', 'service_s', 'ttft_p50', 'read_gb']
out = {'settings': {}, 'vs_ref': {}, 'ref': ref}
for name, runs in S.items():
    m = {k: st.mean(r[k] for r in runs) for k in METRICS if all(r.get(k) is not None for r in runs)}
    m['decode_aa_pct'] = 100 * (max(r['decode_tps'] for r in runs) - min(r['decode_tps'] for r in runs)) / m['decode_tps']
    if all('blocks' in r for r in runs): m['blocks'] = [st.mean(r['blocks'][i] for r in runs) for i in range(len(BLOCKS))]
    out['settings'][name] = {'runs': len(runs), 'source': runs[0]['source'], **m}
    print(f"{name:10s} runs {len(runs)}  decode {m['decode_tps']:.2f} tok/s (A/A {m['decode_aa_pct']:.2f}%, {runs[0]['source']})  "
          f"prompt {m.get('prompt_tps', float('nan')):.1f}  service {m['service_s']:.0f} s  TTFT p50 {m['ttft_p50']:.2f} s"
          + (f"  storage read {m['read_gb']:.0f} GB" if 'read_gb' in m else '')
          + ('  blocks ' + ' / '.join(f'{x:.1f}' for x in m['blocks']) if 'blocks' in m else ''))
for name, runs in S.items():
    if name == ref: continue
    res = {}
    for k in METRICS:
        x = [r[k] for r in S[ref]]; y = [r[k] for r in runs]
        if None in x or None in y or any(isinstance(v, float) and math.isnan(v) for v in x + y): continue
        ch, lo, hi = welch(x, y); res[k] = [ch, lo, hi]
        print(f"{name} vs {ref} {k}: {ch:+.2f}%" + (f" (95% CI {lo:+.2f} .. {hi:+.2f})" if lo is not None else ''))
    out['vs_ref'][name] = res
json.dump(out, sys.stdout if False else open('summary.json', 'w'), indent=1)
print('wrote summary.json')
