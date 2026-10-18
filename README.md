# STRATA_RAM_ADAPT replay kit

Frozen chat requests and two scripts to compare `STRATA_RAM_ADAPT` off / on / off on your own machine. Only the
client-side numbers are used (time to first token, total time, the server's `usage` block); no engine changes needed.

## Request sets (`requests/`)
| file | requests | what |
| --- | ---: | --- |
| `official15.json` | 15 | the bench suite's requests, captured once and frozen (4 warm-up, 3 chat, 4K, 16K, 6 agent) |
| `sharegpt120.json` | 120 | built locally (not redistributed): 30 English ShareGPT conversations × first 4 turns, in order; history = the recorded replies |
| `english47.json` | 47 | authored English session: 6 topics, 29 user/tool-loop turns + 18 background tasks, 38 tools |

`sharegpt120.json` is not included (the dataset's text is not ours to redistribute); `sharegpt_build.py` builds it from
[anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
(revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`, `ShareGPT_V3_unfiltered_cleaned_split.json`, sha256 `35f0e213…`):
first 4 human/gpt pairs alternate, every message ≥97% ASCII and ≤4000 chars, history ≤24000 chars, sorted by id,
shuffled with seed 0, first 30. Recorded replies are used as history so every run sees identical inputs.

    python3 sharegpt_build.py --source ShareGPT_V3_unfiltered_cleaned_split.json --out requests/sharegpt120.json
    sha256sum requests/sharegpt120.json   # 5cd9da0827e6733c083d7a3abaec144a3516f00e2f28fae5728e2866d14c5859

## Run
1. Start the server three times, fresh each time, with the same flags; only the environment differs:
   OFF = no `STRATA_RAM_ADAPT`, ON = `STRATA_RAM_ADAPT=1`. Order: OFF, ON, OFF (ON is judged against the closing OFF).
2. For each start: `python3 replay.py --url http://127.0.0.1:8095 --requests requests/sharegpt120.json --out runs/sg-<arm>.jsonl`
3. `python3 summarize.py runs/sg-off-a.jsonl runs/sg-on.jsonl runs/sg-off-b.jsonl`

It prints decode tok/s, TPOT and TTFT p50/p95, a paired per-request decode-time ratio with a bootstrap 95% interval,
and the cumulative service time with the request from which ON stays ahead. Output lengths can differ between runs, so
compare rates, not totals (the cumulative figure uses the closing OFF run's token counts).

## What we ran (for reference)
RTX 3060 12 GB, 30 GB RAM, Samsung NVMe, Qwen3.8-Flash-Next GSQ IQ2_XS, Strata v0.1.45 + the PR commit.

    strata --serve --prefill auto:16384 --spec 3 --mtp-max-t 3 --spec-min-p 0.5 --suffix-draft 0 --max-context 131072
      --kv k8v4 --pool-workers 7 --ple-io direct --greedy --stop-eos --stats --mmap-experts --mtp <mtp dir>
      --prompt-cache-root 1024 --prompt-cache-every 0 --conversation-cache-min-free-mib 1280 --tail-role-token 8678
      --conversation-cache-min-tokens 3072 --pack <pack> --native <IQ2_XS gguf> --expert-profile <stock profile>
      --resident-budget-gib 20.5 --expert-cache 4200 --vram-reserve-mib 768 --adapt-async 1
      --conversation-cache-mib 1536 --conversation-cache-slots 3 --prompt-cache 2
    env: STRATA_UNBUFFERED_LOAD=1 STRATA_STAGER_THREADS=8 STRATA_STAGER_RING=32 STRATA_LOOKAHEAD=1
         STRATA_PREFILL_CPU_SHARE=auto STRATA_RESIDENT_HEADROOM_GIB=0 STRATA_ADAPT_OVERLAP=0 STRATA_ROUTE_TAIL_SKIP=7
         CUDA_MODULE_LOADING=LAZY; ON adds STRATA_RAM_ADAPT=1

The cache got 3769 of the 4200 slots; the RAM budget holds ~74% of the experts outside the GPU. Results (ON vs closing OFF):
ShareGPT120 decode +33.8% (paired ratio 0.747, 95% CI 0.734–0.761), cumulative time −23%; English47 +19.2% (paired 0.839, 0.819–0.861); official15 +3.2%, prefill unchanged.

## Our raw numbers
`results/*.csv`: one row per request and run (off_before, on, off_after) from the runs above, taken from the engine's
own statistics: generated tokens, prompt/decode ms, TTFT, file/RAM expert reads, SSD bytes, RAM moves (and how many
came out of a stage buffer). No request text. `charts/sharegpt120-cumulative.png` is drawn from `results/sharegpt120.csv`.
