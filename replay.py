#!/usr/bin/env python3
"""Replay frozen chat requests in order against a running `strata --serve` and record client-side timings.

  python3 replay.py --url http://127.0.0.1:8080 --requests requests/sharegpt120.json --out runs/sharegpt-off-a.jsonl

One request at a time, streaming, exactly the stored bodies (only "model" can be overridden). Per request it records
time to first token, total time and the server's usage block; nothing else is needed from the engine.
Standard library only.
"""
import argparse, hashlib, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8080')
ap.add_argument('--requests', required=True, help='JSON with a "requests" list of {"body": ...}')
ap.add_argument('--out', required=True)
ap.add_argument('--model', help='override the "model" field of every body')
ap.add_argument('--header', action='append', default=[], help='extra request header NAME=VALUE (repeatable)')
ap.add_argument('--trace-header', help='optional header carrying a per-request id <trace-prefix><index>')
ap.add_argument('--trace-prefix', default='')
a = ap.parse_args()
reqs = json.load(open(a.requests))['requests']
with open(a.out, 'w') as out:
    for i, r in enumerate(reqs):
        body = dict(r['body'], stream=True, stream_options={'include_usage': True})
        if a.model: body['model'] = a.model
        h = hashlib.sha256(json.dumps(r['body'], sort_keys=True).encode()).hexdigest()
        headers = {'Content-Type': 'application/json', **dict(h.split('=', 1) for h in a.header)}
        if a.trace_header: headers[a.trace_header] = f'{a.trace_prefix}{i:03d}'
        req = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', json.dumps(body).encode(), headers)
        t0 = time.monotonic(); first = None; usage = None; finish = None; chars = 0
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:') or line == 'data: [DONE]': continue
                ev = json.loads(line[5:])
                if ev.get('usage'): usage = ev['usage']
                for ch in ev.get('choices') or []:
                    d = ch.get('delta') or {}
                    piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                    if (piece or d.get('tool_calls')) and first is None: first = time.monotonic()
                    chars += len(piece)
                    if ch.get('finish_reason'): finish = ch['finish_reason']
        t1 = time.monotonic()
        row = {'i': i, 'body_sha256': h, 'ttft_s': (first or t1) - t0, 'total_s': t1 - t0, 'finish': finish,
               'prompt_tokens': (usage or {}).get('prompt_tokens'), 'completion_tokens': (usage or {}).get('completion_tokens'),
               'chars': chars, **{k: r[k] for k in ('conv', 'turn', 'phase', 'scenario') if k in r}}
        out.write(json.dumps(row) + '\n'); out.flush()
        print(f"{i + 1}/{len(reqs)} ttft {row['ttft_s']:.2f}s total {row['total_s']:.1f}s tokens {row['completion_tokens']}", flush=True)
