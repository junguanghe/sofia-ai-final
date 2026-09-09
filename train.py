"""Train one tiny CPU model. Evaluation uses a separate random generator."""
import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from model import Config, TinyGPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="results")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    path = Path(args.checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
    raw = Path("data/input.txt").read_bytes()
    text = raw.decode("utf-8")
    chars = "".join(sorted(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    datasets = {"train": data[:split], "val": data[split:]}
    config = Config(vocab_size=len(chars))
    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_rng = torch.Generator().manual_seed(args.seed + 1)

    def batch(which, rng):
        seq = datasets[which]
        starts = torch.randint(len(seq) - config.context, (8,), generator=rng)
        indices = starts[:, None] + torch.arange(config.context)
        return seq[indices], seq[indices + 1]

    @torch.inference_mode()
    def evaluate():
        model.eval()
        losses = {}
        for which in datasets:
            rng = torch.Generator().manual_seed(2026)  # Same held-out windows each time.
            values = []
            for _ in range(20):
                x, y = batch(which, rng)
                logits, _ = model(x)
                values.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
            losses[which] = sum(values) / len(values)
        model.train()
        return losses

    rows, best = [], float("inf")
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step % 300 == 0 or step == args.steps:
            losses = evaluate()
            row = {"step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                   "elapsed_s": time.perf_counter() - start}
            rows.append(row)
            print(f"step {step:4d}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                  f"elapsed {row['elapsed_s']:.1f}s", flush=True)
            if losses["val"] < best:
                best = losses["val"]
                torch.save({"config": asdict(config), "state_dict": model.state_dict(),
                            "chars": chars, "step": step, "val_loss": best,
                            "seed": args.seed}, path)
        if step == args.steps:
            break
        x, y = batch("train", train_rng)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
        optimizer.step()
    with (output / "training.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    metadata = {"config": asdict(config), "steps": args.steps, "batch_size": 8,
                "learning_rate": 3e-4, "optimizer": "AdamW (PyTorch defaults except lr)",
                "seed": args.seed, "threads": args.threads, "torch": str(torch.__version__),
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_chars": split, "val_chars": len(data) - split,
                "data_sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "elapsed_s": time.perf_counter() - start,
                "best_val_loss": best, "checkpoint": str(path)}
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
