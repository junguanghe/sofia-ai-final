# Sources and reading scope

## Papers

1. Ashish Vaswani et al. (2017). **Attention Is All You Need.** NeurIPS.
   [Paper](https://arxiv.org/abs/1706.03762).
   Read section 3.2 for scaled dot-product and multi-head attention, and section
   3.5 for positions. This project uses a decoder-only architecture with learned
   absolute positions, following the lecture. It does not reproduce the paper's
   translation experiments or full encoder-decoder architecture.

2. Woosuk Kwon et al. (2023). **Efficient Memory Management for Large Language
   Model Serving with PagedAttention.** SOSP.
   [Paper](https://arxiv.org/abs/2309.06180).
   Read sections 2.1–2.2 for prompt processing and autoregressive generation,
   and the beginning of section 3 for cache memory pressure. PagedAttention
   manages cache storage in blocks. This project implements ordinary per-request
   caching with contiguous PyTorch tensors, so its measurements concern reuse,
   not paged allocation or serving throughput.

These are two targeted readings, not an exhaustive literature survey. The
project's contribution is an implementation and controlled demonstration of
an existing technique, not a new research method.

## Code reference

- Andrej Karpathy, **Let's build GPT: from scratch, in code, spelled out.**
  [Lecture video](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- [Pinned lecture implementation](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py).
  Commit `52201428ed7b46804849dea0b3ccf0de9df1a5c3`.
- Reference concepts retained: character vocabulary, individual attention heads,
  causal masking, learned position embeddings, pre-LayerNorm residual blocks,
  ReLU feed-forward network, and autoregressive generation.
- Project changes: smaller configuration, request-local KV input/output,
  offset-aware causal masks, explicit context limit, checkpointing, controlled
  validation sampling, equivalence tests and CPU measurement scripts.
- The upstream source was consulted, not vendored. Its repository did not expose
  a `LICENSE` file at the checked revision. No upstream license is asserted here.

## Data and tools

- [Tiny Shakespeare provenance and checksum](../data/README.md).
- [PyTorch](https://pytorch.org/): tensor operations, autograd and optimization.
- [Matplotlib](https://matplotlib.org/): plots from the recorded measurements.
- [ReportLab](https://www.reportlab.com/): PDF rendering from Markdown source.

Experiment-specific facts come from `results/training.json`,
`results/correctness.json`, `results/benchmark.json`, and the corresponding CSVs.
Code and experiment setup were prepared with Codex assistance. No paper's reported
speedup is presented as a measurement from this project.
