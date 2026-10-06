# Project learning resources

This is a curated reference list for the inference project. It is deliberately
short: use the resource attached to the current task instead of trying to
consume everything in order.

## How to use this list

For each topic:

1. use one visual or conceptual explanation to form a mental model;
2. use one implementation or official reference to settle exact details;
3. return to this repository and produce a trace, test, or measurement;
4. explain the result without the source open.

Reading is preparation. The project artifact is the evidence that the concept
was learned.

## Primary handbook

### Alisa's Book of LLMs

- Link: [Alisa's Book of LLMs](https://alisawuffles.notion.site/alisa-s-book-of-llms)
- Role: broad reference handbook for the project.
- Most relevant now: neural-network tensor conventions, the modern Transformer,
  attention, inference, KV caching, FLOPs, inference memory, GPU execution, and
  numerical precision.
- Use later: scaling, parallelism, mixture-of-experts models, post-training, and
  other architectures.
- Do not read it front to back before continuing. Look up the section connected
  to the current experiment, work through its equations, and then close it and
  reproduce the explanation yourself.

## Primary inference-engineering overview

### *Inference Engineering* by Philip Kiely

- Official edition: [read the book online through Baseten Books](https://www.baseten.co/inference-engineering/book/)
- Publication: Baseten Books, 2026.
- Role: the project-wide map from model execution to GPU hardware, inference
  software, optimization techniques, and production serving.
- Read now:
  - Chapter 1.4, “Measuring Latency and Throughput,” when defining metrics;
  - Chapter 2.2, “LLM Inference Mechanics,” after tracing the local GPT-2 code;
  - Chapter 2.4, “Calculating Inference Bottlenecks,” when beginning the cost
    model;
  - Chapter 4.5, “Performance Benchmarking and Load Testing,” before building
    the benchmark harness.
- Read later:
  - Chapter 3 for GPU architecture and memory hierarchy;
  - Chapter 4 for CUDA, frameworks, inference engines, and profiling;
  - Chapter 5 for quantization, speculative decoding, caching, parallelism, and
    disaggregation;
  - Chapter 7 when the project reaches serving and production concerns.
- Do not use it as a substitute for implementing the small mechanisms in this
  repository. Its value right now is showing where the mechanism fits in the
  larger inference stack.
- The local PDF is copyrighted and is intentionally not committed to this
  repository. The official online edition is the shareable project reference.

## Current phase: architecture to naive inference

### Attention intuition

- [Attention in transformers, step by step — 3Blue1Brown](https://www.3blue1brown.com/lessons/attention/)
- Use for: the meaning of queries, keys, values, dot-product scores, softmax,
  and multiple heads.
- Stop when: you can narrate what information one token is requesting, what
  another token advertises, and what information is transferred.

### Attention and Transformer code

- [Let's build GPT from scratch — Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Watch now: roughly `1:02:00–1:37:50`.
  - `1:02:00`: lower-triangular weighted aggregation;
  - `1:07:11`: queries, keys, values, masking, and scaled dot-product attention;
  - `1:22:01`: multiple heads and the feed-forward network;
  - `1:25:01`: residual connections and LayerNorm.
- Use for: seeing the equations become explicit PyTorch tensor operations.
- Skip for now: the earlier training loop and the later scaling/training
  discussion unless a gap requires them.

### GPT-2-specific visual cross-check

- [The Illustrated GPT-2 — Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
- Use for: masked self-attention in a decoder-only model and the repeated
  next-token loop.
- Caution: treat it as a visual explanation, then resolve exact shapes against
  this repository's code.

### Exact attention operator

- [PyTorch scaled dot-product attention tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch `scaled_dot_product_attention` reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- Use after tracing attention manually: compare the explicit implementation in
  `CausalSelfAttention.forward` with the fused library operation.
- Do not replace the explicit code merely to make it faster yet; first learn
  which operations the fused function combines.

## Generation and pretrained GPT-2

### GPT-2 model contract

- [Hugging Face GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
- Use for: input and output shapes, logits, `past_key_values`, position IDs, and
  the contract for passing only unprocessed tokens when a cache is present.
- Use the Hugging Face implementation as the trusted correctness reference,
  not as the code to copy wholesale.

### Decoding choices

- [Hugging Face text-generation guide](https://huggingface.co/docs/transformers/llm_tutorial)
- Use for: greedy decoding versus sampling and the arguments that control each.
- For correctness work, start with greedy decoding so repeated runs are
  deterministic.

## KV caching

- [Hugging Face caching explanation](https://huggingface.co/docs/transformers/cache_explanation)
- [Hugging Face cache strategies](https://huggingface.co/docs/transformers/kv_cache)
- Read in this order: the explanation first, then the strategy comparison.
- Use for: per-layer key/value shapes, cache growth, attention-mask length, and
  the difference between dynamic and preallocated caches.
- Implement only a simple dynamic cache in this project first. Static,
  offloaded, quantized, and paged caches belong later.

## Benchmarking

- [PyTorch benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- Use for: warmup, repeated measurements, representative thread settings, and
  accelerator synchronization.
- First project metrics: prefill latency, time to first token, per-token decode
  latency, tokens per second, and peak memory when available.

## Primary papers for later

These are references to return to after the corresponding naive implementation
exists:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original
  Transformer equations and multi-head formulation.
- [FlashAttention](https://arxiv.org/abs/2205.14135) — attention as an I/O
  problem; read after learning GPU memory hierarchy.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
  — paged KV-cache management; read after implementing a contiguous cache.

## Resource selection rule

If two sources explain the same concept, do not automatically study both in
full. Use the clearest source for intuition, then consult the more exact source
only where the implementation or equations remain unclear.
