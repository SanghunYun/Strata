#!/usr/bin/env python3
"""Fixed ShareGPT long-session workload. Recorded assistant replies are the history (identical inputs in every arm).
Source: anon8231489123/ShareGPT_Vicuna_unfiltered rev 192ab218, ShareGPT_V3_unfiltered_cleaned_split.json sha256 35f0e213."""
import argparse, hashlib, json, random
from pathlib import Path
ap = argparse.ArgumentParser(description='download: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered '
                             '(revision 192ab2185289094fc556ec8ce5ce1e8e587154ca)')
ap.add_argument('--source', required=True, help='path to ShareGPT_V3_unfiltered_cleaned_split.json')
ap.add_argument('--out', default='requests/sharegpt120.json')
args = ap.parse_args()
SRC = Path(args.source)
SEED, N_CONV, TURNS, MAX_TOKENS = 0, 30, 4, 256
MAX_MSG_CHARS, MAX_HISTORY_CHARS = 4000, 24000

def english(t):
    return t and sum(c.isascii() for c in t) >= 0.97 * len(t)

raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4', 'unexpected dataset file'
data = json.loads(raw)
ok = []
for c in data:
    m = c.get('conversations') or []
    if len(m) < 2 * TURNS: continue
    m = m[:2 * TURNS]
    if [x.get('from') for x in m] != ['human', 'gpt'] * TURNS: continue
    if not all(english(x.get('value', '')) and 1 <= len(x['value']) <= MAX_MSG_CHARS for x in m): continue
    if sum(len(x['value']) for x in m[:-1]) > MAX_HISTORY_CHARS: continue
    ok.append(c)
ok.sort(key=lambda c: c['id'])
random.Random(SEED).shuffle(ok)
pick = ok[:N_CONV]
reqs = []
for k, c in enumerate(pick):
    m = c['conversations'][:2 * TURNS]
    for t in range(TURNS):
        hist = [{'role': 'user' if x['from'] == 'human' else 'assistant', 'content': x['value']} for x in m[:2 * t + 1]]
        reqs.append({'i': len(reqs), 'conv': k, 'conv_id': c['id'], 'turn': t, 'phase': f'turn{t}',
                     'recorded_reply_chars': len(m[2 * t + 1]['value']),
                     'body': {'model': 'strata', 'stream': True, 'max_tokens': MAX_TOKENS, 'temperature': 0,
                              'messages': hist, 'reasoning_effort': 'none',
                              'chat_template_kwargs': {'enable_thinking': False}}})
out = {'source': 'ShareGPT_V3_unfiltered_cleaned_split.json', 'source_sha256': '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4',
       'revision': '192ab2185289094fc556ec8ce5ce1e8e587154ca', 'seed': SEED, 'eligible_conversations': len(ok),
       'conversations': [c['id'] for c in pick], 'turns_per_conversation': TURNS, 'max_tokens': MAX_TOKENS,
       'filter': f'first {TURNS} human/gpt pairs alternate; every message >=97% ASCII and 1..{MAX_MSG_CHARS} chars; '
                 f'history chars <= {MAX_HISTORY_CHARS}; sorted by id, shuffled with seed {SEED}',
       'history': 'recorded ShareGPT assistant replies (frozen); model outputs never enter later inputs',
       'requests': reqs}
blob = json.dumps(out, indent=1, ensure_ascii=False)
Path(args.out).write_text(blob + '\n')
print(len(ok), 'eligible;', len(reqs), 'requests; sha256', hashlib.sha256(blob.encode()).hexdigest()[:16])
print('history chars per request: mean', sum(sum(len(x['content']) for x in r['body']['messages']) for r in reqs) // len(reqs),
      'max', max(sum(len(x['content']) for x in r['body']['messages']) for r in reqs))
