#!/usr/bin/env python3
"""Replay frozen chat requests in order against a running Strata server and record per-request timings.

  python3 replay.py --url http://127.0.0.1:8080 --requests requests/sharegpt120.json --out runs/sharegpt-pr-1.jsonl
  python3 replay.py --url http://127.0.0.1:8080 --standard <Strata>/bench/prompts/standard-v1 --out runs/standard-pr-1.jsonl

One request at a time, streaming, exactly the stored bodies (only "model" can be overridden). Per request it records
time to first token, total time, the server's usage block and the engine's own `timings` (prompt_n / prompt_ms /
predicted_n / predicted_ms, sent by serve/server.py in the last chunk). --standard builds the bodies the way
`strata bench run` does (strata_cli/bench_standard.py payload_of, v0.2.1) in the protocol's order: warm-up, performance
x3, sanity; the optional concurrency stage is skipped. --engine-pid records the engine's /proc/<pid>/io before and after
(read_bytes = what the engine caused to be read from storage) into <out>.meta.json. Standard library only.
"""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8080')
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument('--requests', help='JSON with a "requests" list of {"body": ...}')
src.add_argument('--standard', help='bench/prompts/standard-v1 directory of a Strata checkout')
ap.add_argument('--out', required=True)
ap.add_argument('--model', help='override the "model" field of every body')
ap.add_argument('--engine-pid', type=int, help='record /proc/<pid>/io of the engine process before and after')
ap.add_argument('--header', action='append', default=[], help='extra request header NAME=VALUE (repeatable)')
ap.add_argument('--trace-header', help='optional header carrying a per-request id <trace-prefix><index>')
ap.add_argument('--trace-prefix', default='')
a = ap.parse_args()

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0, "reasoning_effort": "xhigh"}
if a.standard:
    d = Path(a.standard); reqs = []
    for rid in [r['id'] for r in json.loads((d / 'build.json').read_text())['requests']]:
        s = json.loads((d / 'requests' / f'{rid}.json').read_text())
        if s['phase'] == 'concurrency': continue
        body = {"model": "strata", "messages": s["messages"], "stream": True, "stream_options": {"include_usage": True},
                "max_tokens": s["max_output_tokens"], "seed": s["seed"], **SAMPLING, "reasoning_budget_tokens": 0,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
        if s.get("tools"): body["tools"] = s["tools"]; body["tool_choice"] = s.get("tool_choice") or "auto"
        reqs.append({'body': body, 'phase': s['phase'], 'case': rid})
else:
    reqs = json.load(open(a.requests))['requests']

def pio():
    if not a.engine_pid: return None
    return {k: int(v) for k, v in (l.split(': ') for l in Path(f'/proc/{a.engine_pid}/io').read_text().splitlines())}

meta = {'io_start': pio(), 't_start': time.time()}
with open(a.out, 'w') as out:
    for i, r in enumerate(reqs):
        body = dict(r['body'], stream=True, stream_options={'include_usage': True})
        if a.model: body['model'] = a.model
        h = hashlib.sha256(json.dumps(r['body'], sort_keys=True).encode()).hexdigest()
        headers = {'Content-Type': 'application/json', **dict(x.split('=', 1) for x in a.header)}
        if a.trace_header: headers[a.trace_header] = f'{a.trace_prefix}{i:03d}'
        req = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', json.dumps(body).encode(), headers)
        t0 = time.monotonic(); first = None; usage = None; timings = None; finish = None; chars = 0
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:') or line == 'data: [DONE]': continue
                ev = json.loads(line[5:])
                if ev.get('usage'): usage = ev['usage']
                if isinstance(ev.get('timings'), dict): timings = ev['timings']
                for ch in ev.get('choices') or []:
                    dl = ch.get('delta') or {}
                    piece = (dl.get('content') or '') + (dl.get('reasoning_content') or '')
                    if (piece or dl.get('tool_calls')) and first is None: first = time.monotonic()
                    chars += len(piece)
                    if ch.get('finish_reason'): finish = ch['finish_reason']
        t1 = time.monotonic()
        row = {'i': i, 'body_sha256': h, 'ttft_s': (first or t1) - t0, 'total_s': t1 - t0, 'finish': finish,
               'prompt_tokens': (usage or {}).get('prompt_tokens'), 'completion_tokens': (usage or {}).get('completion_tokens'),
               'timings': timings, 'chars': chars, **{k: r[k] for k in ('conv', 'turn', 'phase', 'scenario', 'case') if k in r}}
        out.write(json.dumps(row) + '\n'); out.flush()
        print(f"{i + 1}/{len(reqs)} ttft {row['ttft_s']:.2f}s total {row['total_s']:.1f}s tokens {row['completion_tokens']}", flush=True)
meta.update(io_end=pio(), t_end=time.time(), requests=len(reqs))
Path(a.out + '.meta.json').write_text(json.dumps(meta, indent=1) + '\n')
