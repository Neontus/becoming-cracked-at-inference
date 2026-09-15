# LLM inference systems roadmap

The sequence is concept → derive → implement a toy version → test → benchmark →
profile → explain → inspect a production implementation.

The target pace is roughly 10–15 focused hours per week. Move faster or slower
as needed, but do not skip the build and explanation criteria.

## Phase 1 — Understand the computation

- [ ] Weeks 1–2: implement a tiny Llama-style transformer, train it on a small
  dataset, write a deliberately naive autoregressive generation loop, and trace
  every tensor shape through attention, RoPE, RMSNorm, the MLP, and logits.
- [ ] Week 3: build a calculator for parameter memory, KV-cache memory, FLOPs,
  bytes moved, and arithmetic intensity for prefill and decode.

Exit test: explain why naive generation repeats work, predict how the model's
memory use changes with sequence length, and check both against measurements.

## Phase 2 — Learn GPU performance

- [ ] Week 4: GPU execution and memory hierarchy—SMs, warps, blocks, registers,
  shared memory, HBM, coalescing, occupancy, and latency hiding.
- [ ] Weeks 5–6: implement and optimize softmax, RMSNorm, and matrix
  multiplication in Triton; compare each with a PyTorch reference.
- [ ] Week 7: derive the I/O cost of ordinary attention and implement a
  simplified tiled/online-softmax attention kernel.

Exit test: classify a kernel as compute-, memory-, launch-, or host-bound using
a prediction and profiler evidence—not timing alone.

## Phase 3 — Build an inference engine

- [ ] Week 8: single-request prefill and decode with a correct KV cache.
- [ ] Week 9: request lifecycle and continuous batching scheduler.
- [ ] Week 10: block allocator and paged KV cache.
- [ ] Week 11: shared-prefix lookup, hashing, and eviction policy.
- [ ] Week 12: use Nsight Systems/Compute to find and remove one measured
  bottleneck without breaking correctness.

Exit test: report TTFT, TPOT, throughput, GPU utilization, and memory use under a
defined workload; explain one bottleneck and show the evidence behind the fix.

## Phase 4 — Scale and specialize

- [ ] Week 13: implement small send/receive and collective exercises on two
  GPUs and account for communication bytes.
- [ ] Week 14: implement row- and column-parallel linear layers and integrate a
  two-GPU tensor-parallel path.
- [ ] Week 15: choose one depth topic—speculative decoding, CUDA graphs,
  quantization, chunked prefill/disaggregation, or MoE inference.
- [ ] Week 16: compare the engine with vLLM and SGLang, then trace one production
  subsystem that solves a problem encountered in this project.

Exit test: publish a reproducible technical report and tag the matching code.

## Suggested public notes

Publish when the result exists; dates are intentionally not assigned.

- Why naive autoregressive decoding repeats so much work
- KV caching from first principles
- Predicting prefill and decode performance
- What my first Triton kernel taught me about GPU memory
- FlashAttention as an I/O problem
- Building continuous batching
- From contiguous to paged KV caches
- Finding a real bottleneck with Nsight
- Tensor parallelism and the cost of communication

