# STRATA_RAM_ADAPT replay kit (v0.2.1)

Replays frozen chat requests against a running `serve/server.py` and compares settings from repeated fresh runs.
Speeds come from the engine's own `timings` in the last stream chunk, so no engine changes or extra logging are needed.
The v0.1.45 kit and its results are in [`v0.1.45/`](v0.1.45/).

## Request sets
| set | requests | what |
| --- | ---: | --- |
| `requests/sharegpt120.json` | 120 | continuous use: 30 English ShareGPT conversations × first 4 turns, in order; history = the recorded replies |
| `standard-v1` | 22 | the repository's `bench/prompts/standard-v1` (warm-up, performance ×3, sanity), read straight from your Strata checkout; the optional concurrency stage is skipped |

`sharegpt120.json` is not redistributed; build it from
[anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
(revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`, `ShareGPT_V3_unfiltered_cleaned_split.json`):

    python3 sharegpt_build.py --source ShareGPT_V3_unfiltered_cleaned_split.json --out requests/sharegpt120.json
    sha256sum requests/sharegpt120.json   # 5cd9da0827e6733c083d7a3abaec144a3516f00e2f28fae5728e2866d14c5859

`replay.py --standard` builds the `standard-v1` bodies exactly as `strata bench run` does (v0.2.1 `payload_of`).

## Protocol we used
Settings, each in a fresh engine, identical flags and environment except:

| name | binary | environment |
| --- | --- | --- |
| `off` | v0.2.1 | none |
| `ram-adapt-8` | v0.2.1 | `STRATA_RAM_ADAPT=8` |
| `ram-adapt2` | v0.2.1 | `STRATA_RAM_ADAPT=8 STRATA_RAM_ADAPT2=1` |
| `pr` | v0.2.1 + the PR commit | `STRATA_RAM_ADAPT=1` |

Three runs per setting and request set, interleaved: off, a8, a2, pr | pr, a2, a8, off | a2, off, pr, a8. Before every
run: stop the server, `python3 drop_cache.py <pack dir> <gguf dir>`, wait 60 s, start the server with the setting, then

    python3 replay.py --url http://127.0.0.1:8080 --requests requests/sharegpt120.json --engine-pid <strata pid> \
        --out runs/sharegpt-<setting>-<n>.jsonl
    python3 replay.py --url http://127.0.0.1:8080 --standard <Strata>/bench/prompts/standard-v1 --engine-pid <strata pid> \
        --out runs/standard-<setting>-<n>.jsonl

Use the same expert profile file in every run and do not let the engine save it (`--expert-profile-save` off), so no
run starts from a placement an earlier run learned. Then

    python3 summarize.py --ref ram-adapt-8 off='runs/sharegpt-off-*.jsonl' ram-adapt-8='runs/sharegpt-ram-adapt-8-*.jsonl' \
        ram-adapt2='runs/sharegpt-ram-adapt2-*.jsonl' pr='runs/sharegpt-pr-*.jsonl'

prints, per setting, the mean of the runs and their A/A range (decode, prompt, session time, TTFT p50, storage read
GB, and for the 120-request set the decode rate by request block 1-10 / 11-30 / 31-60 / 61-120), and every setting
against `--ref` with a Welch 95% interval. It refuses runs whose request bodies differ. Output lengths differ between
runs, so compare rates, not totals.

## What we ran
RTX 3060 12 GB, 32 GB RAM, Samsung PM981 NVMe, Linux, Qwen3.8-Flash-Next GSQ IQ2_XS read in place, v0.2.1 `90767252`.

    strata --serve --prefill auto:16384 --spec 3 --mtp-max-t 3 --spec-min-p 0.5 --suffix-draft 0 --max-context 131072
      --kv k8v4 --pool-workers 7 --ple-io direct --greedy --stop-eos --stats --mmap-experts --mtp <mtp dir>
      --prompt-cache-root 1024 --prompt-cache-every 0 --vision --conversation-cache-min-free-mib 768
      --conversation-cache-min-free-mib 1280 --tail-role-token 8678
      --conversation-cache-min-tokens 3072 --pack <IQ2_XS pack> --native <IQ2_XS gguf> --expert-profile <stock profile>
      --resident-budget-gib 20.5 --expert-cache 4200 --vram-reserve-mib 768 --adapt-async 1
      --conversation-cache-mib 1536 --conversation-cache-slots 3 --prompt-cache 2
    env: STRATA_UNBUFFERED_LOAD=1 STRATA_STAGER_THREADS=8 STRATA_STAGER_RING=32 STRATA_LOOKAHEAD=1
         STRATA_PREFILL_CPU_SHARE=auto STRATA_RESIDENT_HEADROOM_GIB=0 STRATA_GGUF_PTE_BUDGET_MB=23500
         STRATA_KV_RECLAIM_EXPERTS=1 STRATA_RETAIN_TURN_BOUNDARIES=0 STRATA_ADAPT_OVERLAP=0 STRATA_ROUTE_TAIL_SKIP=7
         STRATA_PREFILL_TIMING=1 STRATA_DECODE_TIMING=1 CUDA_MODULE_LOADING=LAZY, plus the setting's variables

Requests went through our serving copy of `serve/server.py` (from v0.1.41, with a local logging wrapper), not v0.2.1's;
the engine ran `--greedy`, so `standard-v1`'s sampling settings did not apply. This is not a submittable standard result.

## Our raw numbers (`results/v0.2.1/`)
One file per run in `replay.py`'s format (`<set>-<setting>-<n>.jsonl`, `run_order` = position in the interleaved order),
plus `.meta.json` with the engine's storage read bytes, and `*-summary.json` from `summarize.py`. Taken from the engine's
own statistics; no request or answer text. `summarize.py` on these files gives:

| ShareGPT 120 | off | ram-adapt-8 | ram-adapt2 | pr |
| --- | ---: | ---: | ---: | ---: |
| decode tok/s | 37.70 | 48.71 | 47.65 | 51.40 |
| prompt tok/s | 117.8 | 137.4 | 134.5 | 141.5 |
| session s | 931 | 743 | 761 | 701 |
| storage read GB | 956 | 408 | 506 | 308 |

| standard-v1 | off | ram-adapt-8 | ram-adapt2 | pr |
| --- | ---: | ---: | ---: | ---: |
| decode tok/s | 42.04 | 47.58 | 45.37 | 48.34 |
| prompt tok/s | 572.1 | 578.3 | 577.2 | 578.2 |

`charts/plot_v021.py` draws `charts/sharegpt120-v021.png` (decode tok/s, prompt tok/s and file reads per token over
10-request windows of the 3 runs per setting) from these files.

pr vs ram-adapt-8: ShareGPT decode +5.52% (95% CI +4.56..+6.49), prompt +2.94% (+1.02..+4.87), storage read −24.5%;
standard-v1 decode +1.60% (−0.96..+4.17), prompt −0.02%.
