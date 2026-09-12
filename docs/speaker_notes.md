# Speaker notes (about 10 minutes)

Use these as prompts, then explain the project in your own words. The ten slide
budgets total 600 seconds, including transitions. Industry examples motivate the
experiment; only the CPU measurements are results from this project.

## Slide 1 - 40 seconds

“I studied one basic inference optimization, KV caching. I used a very small
character-level Transformer so the whole experiment could run on my laptop CPU.
Both versions use the same trained weights. Across the three workloads, cached
model execution was about 1.7 to 2.4 times faster, with less than 256 KiB of final
KV tensors. The main learning objective was understanding what can be reused.”

## Slide 2 - 60 seconds

"Inference efficiency matters because serving models requires expensive
infrastructure. Meta's second-quarter filing projected 130 to 145 billion
dollars in capital expenditure for 2026. This includes AI and its core business;
it is not a bill for language-model inference alone.

There are also concrete framework deployments. AWS uses vLLM-based optimizations
in parts of its model hosting, and LinkedIn uses SGLang for ranking in AI Job
Search and AI People Search. LinkedIn's example scores candidates using prefill;
it does not generate a long response. These examples show why efficient model
execution is a real engineering concern."

Sources: [4: Meta Q2 2026 filing](https://www.sec.gov/Archives/edgar/data/1326801/000162828026050705/meta-20260630.htm),
[5: AWS engineering](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/),
[6: LinkedIn engineering](https://www.linkedin.com/blog/engineering/ai/scaling-llm-based-ranking-systems-with-sglang-at-linkedin).

## Slide 3 - 60 seconds

"A naive generation loop repeatedly processes the same history. KV caching saves
historical keys and values so only the new position needs to be computed. The
trade-off is memory: the cache grows with sequence length and concurrent requests.
At small batch sizes, GPU decoding is often limited by memory bandwidth, and
caching does not eliminate the need to read weights and historical KV.

In 2023, LMSYS reported using 50 percent fewer serving GPUs after switching to
vLLM. That was an improvement from a complete serving system, not just turning
on a cache. My project studies the basic mechanism behind this area: how much
time does reuse save, and how much memory does it cost on a laptop CPU?"

Sources: [7: NVIDIA inference guide](https://developer.nvidia.com/blog/?p=73739),
[8: vLLM deployment report, June 2023](https://vllm.ai/blog/2023-06-20-vllm).

## Slide 4 - 45 seconds

“The Transformer paper provides the attention mechanism and causal masking.
Karpathy's lecture provides the small decoder architecture I used as a reference.
The vLLM paper explains autoregressive generation and why KV cache memory matters
for serving. My implementation uses simple contiguous tensors. PagedAttention
solves a larger memory-management problem that this experiment does not test.”

## Slide 5 - 85 seconds

Walk through the ABC example slowly. The first forward predicts D and saves the
keys and values for ABC in every layer. The next forward accepts D, calculates
its query/key/value and uses cached history to predict E. Explain the formula:
the new query matches all historical keys and combines their values. Historical
queries are unnecessary because only the new position's output is needed.

Emphasize that causal masking protects historical states from future tokens.
Caching saves repeated computation. It does not remove attention over history.

## Slide 6 - 75 seconds

Explain `past`, `torch.cat`, and the cache nesting by layer and head. Mention the
two implementation details that matter most: position offset and mask rows.
Eval mode switches off dropout, while inference mode disables gradient tracking.
The tests compare all-position logits as well as greedy sequences. The observed
small numerical difference comes from different floating-point execution shapes.
It stayed within the specified tolerances on all tested inputs.

## Slide 7 - 75 seconds

“I trained once, on the first 90 percent of Tiny Shakespeare. The final 10 percent
was reserved for validation. The vocabulary has 65 characters. Training took
about five minutes and validation cross-entropy fell to 1.83. This is still a
small imperfect language model. Its writing quality is not the variable in the
cache experiment.”

Explain fixed continuations, warmup, shuffled order and fixed threads. Clarify
that the timing measures forward calls, including cache operations, but excludes
sampling and application overhead. There are 63 decode calls after prefill for
64 output predictions.

## Slide 8 - 45 seconds

Explain both axes and the error bars. Cached decode is roughly flat over this
short length range, while recomputation slows down as more history is processed.
Do not claim cached decode has constant complexity: it still attends over all
keys. At this small size other costs can dominate, and the timings have outliers.

## Slide 9 - 85 seconds

The speedup is the uncached median total divided by the cached median total.
Explain the two factors in the memory formula that are often confused: the
leading 2 means K and V, and head count is already included in model width.
At prompt 64, final cache length is 127, which is 254 KiB. The 128th predicted
position has not yet been processed. Mention cat copies, the fixed window, the
single CPU and the lack of a production-serving comparison.

## Slide 10 - 30 seconds

Point to the repository containing the implementation, measurements, figures and
commands. Credit the architecture reference and AI assistance accurately. End
with the main finding: retaining historical K/V avoided enough recomputation to
speed up this small CPU workload, at a directly measurable memory cost.

## Questions to rehearse

- Does caching change the learned model? No. The same parameters implement the
  same causal computation, subject to floating-point differences.
- Why no Q cache? Only the latest position's attention output is required during
  generation, so its new query is sufficient.
- Is this prefix caching? It is reuse within one request, not sharing prompt
  computation across different requests.
- Why not compare quality scores? Both paths use the same checkpoint. Numerical
  equivalence is the relevant check; generation quality belongs to training.
- Can it generate indefinitely? No. This implementation rejects requests beyond
  128 total characters, because learned absolute positions have a fixed range.
- Does 254 KiB mean the whole process uses that much memory? No. It counts only
  the final K/V tensors, excluding weights and temporary allocations.
- Why use fixed continuations? To hold the input work constant while comparing
  the two forward paths. A separate sample demonstrates actual generation.
- Would the same speedup hold on a GPU? The experiment does not establish that.
  Kernel overhead, parallelism, memory bandwidth and model size all differ.
- Do production frameworks already use KV caching? Yes. This project explains
  that basic mechanism. It does not claim a new improvement over vLLM or SGLang.
- Does KV caching solve the memory bottleneck? It reduces recomputation but
  occupies memory, and historical KV must still be read. Paged allocation,
  batching and other techniques address additional serving problems.
- Does the 50 percent GPU reduction mean ordinary KV caching saves 50 percent?
  No. The 2023 deployment report concerns the whole vLLM serving system. The
  baseline also used caching; the benefit includes better serving and memory
  management. No isolated dollar saving is reported for our technique.
- Can our CPU speedup be applied to Meta's spending? No. Company-wide capital
  investment includes many workloads and fixed assets; our experiment measures
  one small model's execution time, not a company's total cost.

## Optional cost illustration (Q&A only, outside the 10-minute talk)

This is a hypothetical calculation, not a company result or a KV-cache finding.
If a service spends $10 million per year on GPU usage that can scale down, and an
optimization gives 1.25x throughput at the same latency target, an ideal estimate
is $10 million / 1.25 = $8 million, or $2 million saved per year. It assumes fixed
traffic and unit prices, capacity that can shrink proportionally, and no added
costs. If hardware is already owned, the immediate benefit may instead be spare
capacity or postponed purchases. Do not multiply our CPU speedup by a company's
capital expenditure.
