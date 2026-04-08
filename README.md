# autoresearch

![teaser](progress.png)

`autoresearch` is a tiny autonomous LLM research loop. An agent edits [`train.py`](train.py), runs a fixed-length training job, evaluates `val_bpb`, keeps improvements, and discards regressions. This fork is retuned for **Jetson AGX Orin on Jetson Linux R35 / JetPack 5.x** and is meant to be run from a Docker container.

## What changed for Jetson

The original repo was tuned for an H100-class environment. This Jetson port changes the runtime so it is practical on AGX Orin:

- Docker-first workflow via [`Dockerfile.jetson`](Dockerfile.jetson)
- Smaller defaults in [`prepare.py`](prepare.py) and [`train.py`](train.py)
- PyTorch SDPA fallback instead of requiring Hopper-oriented FlashAttention tooling
- `torch.compile` disabled by default
- Byte-level tokenizer fallback enabled by default so the container does not depend on `rustbpe`

The research loop stays the same:

- [`prepare.py`](prepare.py): data prep, tokenizer, dataloader, evaluation
- [`train.py`](train.py): model, optimizer, training loop
- [`program.md`](program.md): instructions for an external coding agent

## Quick Start

Build the container:

```bash
./docker/build-jetson.sh
```

Start a shell in the container:

```bash
./docker/run-jetson.sh
```

Inside the container, prepare the cache and run a baseline experiment:

```bash
python3 prepare.py --num-shards 4
AUTORESEARCH_RUN_DESCRIPTION="jetson orin baseline" python3 train.py
python3 analyze_results.py
```

If everything is healthy, `train.py` will run for about 5 minutes and print a final block like:

```text
---
val_bpb:          ...
training_seconds: ...
total_seconds:    ...
peak_vram_mb:     ...
mfu_percent:      ...
total_tokens_M:   ...
num_steps:        ...
num_params_M:     ...
depth:            ...
```

## Docker Notes

The helper script runs:

- `--runtime nvidia`
- `--network host`
- `--ipc host`
- a bind mount for the repo
- a named Docker volume for `~/.cache/autoresearch`

The image defaults are set in [`Dockerfile.jetson`](Dockerfile.jetson):

- `AUTORESEARCH_TOKENIZER_MODE=byte`
- `AUTORESEARCH_ATTENTION_BACKEND=eager`
- `AUTORESEARCH_AMP_DTYPE=fp32`
- `AUTORESEARCH_OPTIMIZER=adamw`
- `AUTORESEARCH_USE_VALUE_EMBEDS=0`
- `AUTORESEARCH_USE_COMPILE=0`
- `AUTORESEARCH_MAX_SEQ_LEN=512`
- `AUTORESEARCH_EVAL_TOKENS=524288`
- `AUTORESEARCH_VOCAB_SIZE=4096`

These are conservative defaults meant to boot reliably on Orin. You can override them with `docker run -e ...` or by editing the Dockerfile.

The Jetson container intentionally installs only the runtime dependencies needed by [`prepare.py`](prepare.py) and [`train.py`](train.py). Notebook-only analysis dependencies were left out to keep Python 3.8 / `aarch64` resolution simple on Jetson R35.

For terminal-friendly result analysis without Jupyter, use [`analyze_results.py`](analyze_results.py). It reads [`results.tsv`](results.tsv), prints summary stats, and regenerates `autoresearch_progress.png`.

`train.py` now writes two outputs automatically:

- [`run.log`](run.log): full newline-delimited run log, including per-step progress and final summary
- [`results.tsv`](results.tsv): one appended experiment row per completed run

Useful run metadata environment variables:

- `AUTORESEARCH_RUN_DESCRIPTION`: human-readable description stored in `results.tsv`
- `AUTORESEARCH_RESULT_STATUS`: result label stored in `results.tsv` on successful runs; defaults to `keep`
- `AUTORESEARCH_LOG_PATH`: optional custom log file path; defaults to `run.log`
- `AUTORESEARCH_RESULTS_PATH`: optional custom TSV path; defaults to `results.tsv`

## Runtime Controls

Useful environment variables:

- `AUTORESEARCH_MAX_SEQ_LEN`: context length used by prep and training
- `AUTORESEARCH_EVAL_TOKENS`: validation token budget
- `AUTORESEARCH_TIME_BUDGET`: wall-clock training budget in seconds
- `AUTORESEARCH_TOKENIZER_MODE`: `byte`, `rustbpe`, or `auto`
- `AUTORESEARCH_ATTENTION_BACKEND`: `eager`, `sdpa`, `kernel`, or `auto`
- `AUTORESEARCH_AMP_DTYPE`: `fp32`, `fp16`, `bf16`, or `auto`
- `AUTORESEARCH_USE_COMPILE`: `0` or `1`
- `AUTORESEARCH_DEVICE_PEAK_FLOPS`: optional number used for MFU estimation

## Agent Workflow

Once the container is working, you can point a coding agent at [`program.md`](program.md) and let it iterate on [`train.py`](train.py). The intended loop is:

1. Establish a baseline by running the current `train.py`
2. Edit only `train.py`
3. Run another experiment
4. Keep the change if `val_bpb` improves
5. Revert otherwise

## Practical Expectations On Orin

This port is optimized for reliability, not raw speed parity with data-center GPUs. Expect:

- much smaller models
- lower throughput
- more aggressive memory constraints
- slower evaluation

If you want to push harder later, the first knobs to revisit are:

- `DEPTH`
- `DEVICE_BATCH_SIZE`
- `TOTAL_BATCH_SIZE`
- `MAX_SEQ_LEN`
- `AUTORESEARCH_TOKENIZER_MODE`
- `AUTORESEARCH_ATTENTION_BACKEND`

## Project Layout

```text
prepare.py             data prep, tokenizer, dataloader, evaluation
train.py               model, optimizer, training loop
analyze_results.py     terminal analysis + progress plot
program.md             autonomous experiment instructions
Dockerfile.jetson      Jetson AGX Orin container image
requirements-jetson.txt  Python deps installed into the container
docker/build-jetson.sh build helper
docker/run-jetson.sh   run helper
```

## License

MIT



To use it on the Orin:
```bash
cd /home/pi/autoresearch
./docker/build-jetson.sh
./docker/run-jetson.sh
```
Then inside the container:
```bash
python3 prepare.py --num-shards 4
python3 train.py

python3 train.py > run.log 2>&1

python3 analyze_results.py --results results.tsv --plot autoresearch_progress.png
```
