# Benchmark conventions

Inference reports should define at least:

- TTFT: arrival to first generated token;
- TPOT: time per output token after the first token;
- throughput: generated tokens per second and/or completed requests per second;
- latency distribution: median and tail (for example p95 or p99) under serving
  workloads;
- peak allocated/reserved memory and KV-cache capacity;
- workload: prompt/output distributions, concurrency, and arrival pattern.

Run correctness checks before timing. Warm up compiled kernels and caches. Use
explicit device synchronization for isolated GPU timings. Report multiple runs
and a robust statistic rather than the best sample.

