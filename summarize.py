#!/usr/bin/env python3
"""Compare OFF / ON / OFF replays of the same requests (client-side numbers only).

  python3 summarize.py runs/sharegpt-off-a.jsonl runs/sharegpt-on.jsonl runs/sharegpt-off-b.jsonl

ON is judged against the closing OFF run; the two OFF runs show the run-to-run spread.
Decode rate per request = completion tokens / (total time - time to first token).
"""
import json, random, sys

def load(p): return [json.loads(x) for x in open(p)]
def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
def decode_s(r): return r['total_s'] - r['ttft_s']

def summary(rs):
    tok = sum(r['completion_tokens'] for r in rs); dec = sum(decode_s(r) for r in rs)
    tpot = [1000 * decode_s(r) / r['completion_tokens'] for r in rs if r['completion_tokens'] > 1]
    return {'requests': len(rs), 'completion_tokens': tok, 'decode_tok_s': tok / dec,
            'tpot_ms_p50': pct(tpot, .5), 'tpot_ms_p95': pct(tpot, .95),
            'ttft_s_p50': pct([r['ttft_s'] for r in rs], .5), 'ttft_s_p95': pct([r['ttft_s'] for r in rs], .95),
            'total_s': sum(r['total_s'] for r in rs)}

off_a, on, off_b = (load(p) for p in sys.argv[1:4])
assert len(off_a) == len(on) == len(off_b)
assert all(a['body_sha256'] == b['body_sha256'] == c['body_sha256'] for a, b, c in zip(off_a, on, off_b)), 'inputs differ'
S = {name: summary(rs) for name, rs in [('off_open', off_a), ('on', on), ('off_close', off_b)]}
for name, s in S.items():
    print(f"{name:9s} decode {s['decode_tok_s']:.2f} tok/s  TPOT p50/p95 {s['tpot_ms_p50']:.1f}/{s['tpot_ms_p95']:.1f} ms  "
          f"TTFT p50/p95 {s['ttft_s_p50']:.2f}/{s['ttft_s_p95']:.2f} s  total {s['total_s']:.0f} s  tokens {s['completion_tokens']}")
print(f"decode ON vs closing OFF: {100 * (S['on']['decode_tok_s'] / S['off_close']['decode_tok_s'] - 1):+.2f}%   "
      f"(OFF vs OFF: {100 * (S['off_close']['decode_tok_s'] / S['off_open']['decode_tok_s'] - 1):+.2f}%)")
pairs = [(x, y) for x, y in zip(on, off_b) if x['completion_tokens'] > 1 and y['completion_tokens'] > 1]
ratio = lambda ps: sum(decode_s(x) / x['completion_tokens'] * y['completion_tokens'] for x, y in ps) / sum(decode_s(y) for _, y in ps)
rnd = random.Random(0); boot = sorted(ratio([rnd.choice(pairs) for _ in pairs]) for _ in range(2000))
print(f"paired decode time ON/OFF at equal tokens: {ratio(pairs):.3f} (95% CI {boot[50]:.3f}-{boot[1949]:.3f})")
# cumulative service time at the closing OFF run's output lengths (output lengths can differ between runs)
c_on = c_off = 0; curve = []
for x, y in zip(on, off_b):
    per_tok = decode_s(x) / x['completion_tokens'] if x['completion_tokens'] else 0
    c_on += x['ttft_s'] + per_tok * y['completion_tokens']; c_off += y['total_s']; curve.append((c_on, c_off))
be = next((i for i in range(len(curve)) if all(a <= b for a, b in curve[i:])), None)
print(f"cumulative service time: ON {c_on:.0f} s vs OFF {c_off:.0f} s ({100 * (1 - c_on / c_off):+.1f}% saved); "
      f"ON stays ahead from request {None if be is None else be + 1}")
json.dump({'summary': S, 'paired_ratio': ratio(pairs), 'paired_ci95': [boot[50], boot[1949]],
           'cumulative': [{'i': i, 'on_s': a, 'off_s': b} for i, (a, b) in enumerate(curve)], 'break_even_request': be},
          open(sys.argv[3].rsplit('.', 1)[0] + '.summary.json', 'w'), indent=1)
