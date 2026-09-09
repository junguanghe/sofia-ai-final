"""Generate a short sample from the local trained checkpoint."""
import argparse

import torch

from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint("checkpoints/tiny_gpt.pt")
    torch.manual_seed(args.seed)
    chars = checkpoint["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    tokens = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long)
    output = model.generate(tokens, args.tokens, not args.no_cache, args.temperature)
    print("".join(chars[i] for i in output[0].tolist()))


if __name__ == "__main__":
    main()
