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

## Industry background for slides 2-3

Source labels [4]-[8] match the slide citations. These are short, targeted
background readings, checked on September 12, 2026; they add no implementation
or experimental requirements.

- **[4] Meta, Q2 2026 Form 10-Q.**
  [SEC filing](https://www.sec.gov/Archives/edgar/data/1326801/000162828026050705/meta-20260630.htm).
  Search for "$130 billion to $145 billion" in the capital expenditure outlook.
  This is a full-year forecast supporting AI and the core business, including
  principal payments on finance leases. It is not realized full-year spending,
  an inference-only operating cost, or a basis for estimating KV-cache savings.
- **[5] AWS, February 25, 2026. Efficiently serve dozens of fine-tuned models
  with vLLM on Amazon SageMaker AI and Amazon Bedrock.**
  [Engineering article](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/).
  Read the introduction and conclusion for the multi-LoRA serving use case and
  the availability of Amazon-specific optimizations. This establishes a concrete
  vLLM-based hosting example, not that every AWS model uses this backend.
- **[6] LinkedIn, Scaling LLM-Based ranking systems with SGLang at LinkedIn.**
  [Engineering article](https://www.linkedin.com/blog/engineering/ai/scaling-llm-based-ranking-systems-with-sglang-at-linkedin).
  Read "Why prefill-only ranking is different" and "Impact at scale".
  SGLang supports ranking in AI Job Search and AI People Search. This workload
  uses prefill for scoring without iterative decoding. Its reported gains
  combine batching, prefix reuse and runtime changes; they are not our baseline.
- **[7] NVIDIA, Mastering LLM Techniques: Inference Optimization.**
  [Technical guide](https://developer.nvidia.com/blog/?p=73739).
  Read "Understanding LLM inference", "Key-value caching" and "LLM memory
  requirement". Prefill has substantial parallel computation; decode, especially
  at small batches, is often limited by memory bandwidth. Cached K/V still need
  to be read, and cache capacity grows with context and concurrent requests.
  The actual bottleneck depends on the model, batch size, hardware and runtime.
- **[8] vLLM team, June 20, 2023. vLLM: Easy, Fast, and Cheap LLM Serving
  with PagedAttention.**
  [Deployment report](https://vllm.ai/blog/2023-06-20-vllm).
  Read "The Silent Hero Behind LMSYS Vicuna and Chatbot Arena". The team reports
  50% fewer serving GPUs after adopting vLLM, with about 30,000 requests per day
  on average and a peak of 60,000. This is an author-reported historical deployment
  result for the whole serving system, not an isolated cache-on/cache-off test
  or a published dollar saving. The paper listed above reports a separate
  2-4x throughput improvement over FasterTransformer and Orca at similar latency
  in its evaluation; that number is not used as a project result.

Additional scale context, for optional reading rather than the slide narrative:

- [OpenAI / Oracle, July 22, 2025](https://openai.com/index/stargate-advances-with-partnership-with-oracle/):
  4.5 GW of additional capacity; with Abilene, over 5 GW under development,
  expected to run over two million chips.
- [Stargate expansion, September 23, 2025](https://openai.com/index/five-new-stargate-sites/):
  nearly 7 GW of planned capacity and over $400 billion in investment over the
  following three years. This overlaps the July announcement; do not add them.
  These are dated announcements of plans, not verified current commissioned
  capacity. GW measures electrical power capacity, not FLOPS.

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
