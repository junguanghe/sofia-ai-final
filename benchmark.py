"""CPU model execution benchmark: identical held-out token continuations."""
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from model import cache_bytes, load_checkpoint


@torch.inference_mode()
def replay(model, tokens, prompt_length, generated, use_cache):
    start = time.perf_counter()
    _, cache = model(tokens[:, :prompt_length], use_cache=use_cache, last_only=True)
    prefill_end = time.perf_counter()
    # Prefill predicts output #1. The remaining G-1 predictions need G-1 forwards.
    for position in range(prompt_length, prompt_length + generated - 1):
        current = tokens[:, position:position + 1] if use_cache else tokens[:, :position + 1]
        _, cache = model(current, cache, use_cache, last_only=True)
    end = time.perf_counter()
    return {"prefill_ms": 1000 * (prefill_end - start),
            "decode_ms": 1000 * (end - prefill_end), "total_ms": 1000 * (end - start),
            "cache_bytes": cache_bytes(cache) if use_cache else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="results")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("use at least two repeats")
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint)
    chars = checkpoint["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    text = Path("data/input.txt").read_text()
    held_out = text[int(0.9 * len(text)):]
    tokens = torch.tensor([[stoi[c] for c in held_out]], dtype=torch.long)
    jobs = [(p, cache) for p in (16, 32, 64) for cache in (False, True)]
    for p, cache in jobs:
        for _ in range(2):
            replay(model, tokens[:, :128], p, 64, cache)
    rows = []
    rng = random.Random(2026)
    for repeat in range(args.repeats):
        order = jobs.copy(); rng.shuffle(order)
        offset = (repeat * 1000) % (tokens.size(1) - 128)
        sample = tokens[:, offset:offset + 128]
        for p, cache in order:
            row = {"repeat": repeat, "prompt_tokens": p, "output_tokens": 64,
                   "mode": "cached" if cache else "uncached", "val_offset": offset,
                   **replay(model, sample, p, 64, cache)}
            rows.append(row)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    with (output / "benchmark.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = []
    for p, cache in jobs:
        mode = "cached" if cache else "uncached"
        group = [r for r in rows if r["prompt_tokens"] == p and r["mode"] == mode]
        med = {key: statistics.median(r[key] for r in group)
               for key in ("prefill_ms", "decode_ms", "total_ms")}
        summary.append({"prompt_tokens": p, "mode": mode, **med,
                        "decode_chars_s": 63_000 / med["decode_ms"],
                        "total_min_ms": min(r["total_ms"] for r in group),
                        "total_max_ms": max(r["total_ms"] for r in group),
                        "cache_KiB": group[0]["cache_bytes"] / 1024})
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    report = {"utc": datetime.now(timezone.utc).isoformat(), "cpu": cpu,
              "platform": platform.platform(), "python": platform.python_version(),
              "torch": str(torch.__version__), "threads": args.threads,
              "dtype": "float32", "batch_size": 1, "repeats": args.repeats,
              "warmup_runs_per_config": 2, "shuffle_seed": 2026,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "data_sha256": hashlib.sha256(Path("data/input.txt").read_bytes()).hexdigest(),
              "timing": "model forward calls only; excludes loading, tokenization, sampling and text assembly",
              "decode_predictions": 63, "summary": summary}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in summary:
        print(f"prompt={row['prompt_tokens']:2d} {row['mode']:8s}: "
              f"total {row['total_ms']:.1f} ms, decode {row['decode_chars_s']:.1f} chars/s")


if __name__ == "__main__":
    main()
