# October–December 2026 curriculum: toward inference OSS contributions

Last updated: 2026-10-13  
Plan window: 2026-10-13 through 2026-12-31

This is the dated, living plan for moving from the GPT-2 architecture work in
this repository toward useful upstream contributions to an inference engine
such as vLLM or SGLang. The broader sequence remains in
[`ROADMAP.md`](ROADMAP.md); the immediate GPT-2 work remains in
[`INFERENCE_LEARNING_BRIEF.md`](INFERENCE_LEARNING_BRIEF.md).

## Starting point on October 13

I can explain the high-level GPT-2 data path and have implemented its main
modules locally. I have not yet completed pretrained-weight loading, a naive
generation loop, a trustworthy benchmark, KV caching, or GPU profiling.

For *Inference Engineering*, I have read the preface and Chapter 0,
“Inference.” Chapter 1, “Prerequisites,” is next.

This is enough background to begin the plan. It is not yet enough to make a
responsible engine-level performance change: first I need to connect the model
architecture to repeated generation, measurements, memory behavior, and the
structure of a production engine.

Training GPT-2 is not a prerequisite for this goal. I need to understand what
the forward and backward passes mean, but the working path is pretrained
weights plus inference.

## Year-end outcome

By December 31, I want to have:

- a correct local GPT-2 generation path with pretrained weights;
- an uncached baseline and a correct KV-cached implementation;
- a small cost model and reproducible measurements for prefill and decode;
- one GPU or operator-level profiling exercise, on accessible hardware;
- an architecture map of one production engine;
- one primary upstream project, rather than splitting attention between two;
- at least one useful upstream artifact: a reproduced issue, test, benchmark,
  documentation improvement, or code change; and
- ideally, one reviewed or merged code contribution with correctness or
  performance evidence.

A merge is partly controlled by maintainers and review timing, so the committed
goal is an upstream-ready contribution and responsive review follow-through.

### What “notable” means here

A contribution is notable if it solves a real project need, has a clear
reproduction or motivation, includes the right tests or benchmarks, and can be
explained end to end. It does not have to be large. A focused correctness fix
with a regression test is more meaningful than a large speculative rewrite.

The outcome ladder is:

- **Floor:** one high-quality issue reproduction or documentation/test PR and
  a written map of the relevant subsystem.
- **Target:** one reviewed or merged code PR plus a useful issue reproduction,
  test, benchmark, or documentation contribution.
- **Stretch:** a second contribution in the same subsystem or a measured
  performance/correctness improvement.

## Weekly operating rhythm

Plan around 10 focused hours each week:

- 4 hours building and testing in this repository;
- 2 hours reading, with notes tied to the current experiment;
- 2 hours measuring, profiling, or interpreting results;
- 1 hour reading production-engine code, issues, and pull requests; and
- 1 hour writing a session note or concise technical explanation.

If a week is constrained, preserve the build/test work and one explanation.
Reduce reading breadth instead of rushing past a correctness gate.

Every week should end with evidence: code, a passing test, a shape trace, a
measurement table, a profiler capture, an issue reproduction, or a written
subsystem map. Reading alone does not complete a week.

## Dated plan

### October 13–18 — Close the architecture gap

**Concepts**

- Derive every important attention and MLP tensor shape from `B`, `T`, `C`,
  and `H`.
- Explain Q/K/V projections, causal masking, multi-head splitting, the output
  projection, GELU, LayerNorm, and residual additions.
- Explain why attention communicates across token positions while the MLP acts
  independently at each position.

**Build**

- Complete Session 2 in [`INFERENCE_LEARNING_BRIEF.md`](INFERENCE_LEARNING_BRIEF.md).
- Produce a shape trace against `src/inference_lab/gpt2.py` and a more detailed
  Transformer-block diagram.

**Read**

- *Inference Engineering*, Chapter 1, “Prerequisites.”
- Use the attention sources in [`RESOURCES.md`](RESOURCES.md) only where the
  shape trace exposes a gap.

**Gate**

- With references closed, draw the block and recover all key tensor shapes.

### October 19–25 — Make inference real

**Concepts**

- Separate a forward pass, loss, backward pass, optimizer update, prefill, and
  decode.
- Understand what pretrained-weight loading copies and why exact architecture
  compatibility matters.

**Build**

- Complete Session 3: load pretrained GPT-2 weights using trusted glue code.
- Add deterministic greedy generation that passes the entire growing sequence
  through the model on every step.
- Compare selected logits or a short greedy continuation with Hugging Face.

**Read**

- *Inference Engineering*, Chapter 2 through section 2.2, “LLM Inference
  Mechanics.”
- The GPT-2 model contract and greedy-generation references in
  [`RESOURCES.md`](RESOURCES.md).

**Gate**

- Generate text reproducibly and point to exactly what the naive loop
  recomputes.

### October 26–November 1 — Establish the baseline

**Concepts**

- Define time to first token, time per output token, throughput, peak memory,
  warmup, synchronization, and measurement variance.
- Separate prefill measurements from decode measurements.

**Build**

- Complete Session 4 without optimizing the model.
- Add a repeatable benchmark command and record hardware, dtype, batch size,
  prompt length, output length, software versions, and timing method.
- Measure at least a small matrix of prompt and output lengths.

**Read**

- *Inference Engineering*, sections 2.4–2.5 for bottlenecks and optimization
  framing.
- The PyTorch benchmarking reference in [`RESOURCES.md`](RESOURCES.md).

**Gate**

- Rerun the same benchmark and explain why the results are comparable—or why
  they are not.

### November 2–8 — Implement the first inference optimization

**Concepts**

- Explain why old keys and values are reusable, while the new token still
  needs new Q/K/V projections.
- Derive KV-cache shape and memory growth by layer, batch, sequence length,
  number of KV heads, and head dimension.

**Build**

- Complete Session 5: implement the simplest correct dynamic KV cache.
- Assert that cached and uncached greedy decoding choose the same tokens.
- Measure both paths across more than one prompt and generation length.

**Read**

- The two KV-cache references in [`RESOURCES.md`](RESOURCES.md).
- Skim *Inference Engineering* section 5.3 only after the local cache works.

**Gate**

- Explain the speed/memory tradeoff from measurements and predicted cache
  size, not only from intuition.

### November 9–15 — Build a cost model

**Concepts**

- Parameter memory, activation memory, KV-cache memory, approximate FLOPs,
  bytes moved, arithmetic intensity, and compute- versus memory-bound work.
- Why prefill and decode often have different bottlenecks.

**Build**

- Create a small calculator for GPT-2 parameter memory, KV-cache memory, and
  approximate linear/attention work.
- Predict the direction of benchmark results before running them; compare the
  prediction with the data afterward.

**Read**

- Revisit *Inference Engineering* section 2.4.
- Use Alisa's Book of LLMs for FLOPs, inference memory, and arithmetic
  intensity when the calculation needs a second explanation.

**Gate**

- Given model dimensions and a workload, estimate the dominant memory use and
  identify the likely bottleneck with stated assumptions.

### November 16–22 — Learn the GPU model needed for profiling

**Concepts**

- Kernels, threads, warps, blocks, SMs, registers, shared memory, HBM,
  coalescing, occupancy, launch overhead, and latency hiding.
- The difference between a timing result and evidence for a bottleneck.

**Build**

- Profile one inference path or representative operator on available hardware.
- Annotate the trace: CPU launch, kernel execution, transfers, and idle time.
- If CUDA hardware is unavailable, preserve the conceptual/profiling study and
  schedule the CUDA capture on a rented or school GPU rather than pretending a
  CPU trace answers a GPU question.

**Read**

- *Inference Engineering*, Chapter 3, “Hardware.”

**Gate**

- Classify one observed operation as compute-, memory-, launch-, or host-bound
  and state what evidence would falsify that classification.

### November 23–29 — Move from operators to inference engines

**Concepts**

- Follow the request path from API input to tokenization, scheduling, model
  execution, KV allocation, sampling, streaming, and metrics.
- Learn continuous batching, paged KV caches, prefix caching, and the purpose
  of an engine scheduler at an architectural level.

**Build**

- Draw a neutral engine diagram before reading either implementation deeply.
- Run one small engine example if the available machine supports it.
- Record which mechanisms solve problems already encountered in the GPT-2 lab.

**Read**

- *Inference Engineering*, Chapter 4, “Software.”
- The PagedAttention paper after reviewing the local contiguous KV cache.

**Gate**

- Trace a request through the neutral diagram and explain where TTFT, TPOT,
  throughput, and memory pressure arise.

### November 30–December 6 — Choose and map one upstream project

Use both projects for comparison, then select one as the primary contribution
target by December 6. Do not attempt simultaneous first contributions.

**vLLM reconnaissance**

- Read the [vLLM contribution guide](https://docs.vllm.ai/en/latest/contributing/),
  its testing workflow, and its current
  [good first issues](https://github.com/vllm-project/vllm/contribute).
- Read the repository's `AGENTS.md` before any AI-assisted contribution. It
  requires duplicate-work checks, meaningful scope, human understanding of
  every changed line, test evidence, disclosure of AI assistance, and signed
  commits.

**SGLang reconnaissance**

- Read the [SGLang repository](https://github.com/sgl-project/sglang), current
  issues, and the [2026 Q4 roadmap](https://github.com/sgl-project/sglang/issues/42473).
- Compare the roadmap areas—scheduler, KV-cache management, speculative
  decoding, usability, hardware support, and kernels—with the concepts already
  demonstrated in this repository.

**Selection criteria**

- I can build or at least exercise the relevant path on hardware I can access.
- I can run a narrow local test before relying on upstream CI.
- The subsystem connects to something I have implemented or measured.
- The issue has a responsive maintainer or recent related activity.
- No open pull request already duplicates the intended work.
- I can explain the expected behavior and failure before changing code.

**Produce**

- A one-page map of the selected repository: request entry point, scheduler,
  model executor, KV-cache manager, attention backend, sampling path, tests,
  benchmarks, and subsystem owners or recent reviewers.
- A shortlist of three live candidate tasks, with required hardware, likely
  tests, scope, and duplication checks. Live issues change, so do not freeze an
  issue choice months in advance.

**Gate**

- Select one project and one subsystem. Explain why the candidate task is both
  useful upstream and feasible to verify.

### December 7–13 — Reproduce before changing

**OSS work**

- Set up the selected project exactly as its current contributor guide states.
- Run one narrow existing test in the chosen subsystem.
- Reproduce the candidate issue or establish a missing test/documented gap.
- Read recent issues and pull requests touching the same files.
- Comment with a concise reproduction and proposed approach when maintainer
  coordination is appropriate; do not claim an issue before confirming it is
  still open and unclaimed.

**Produce**

- A minimal reproduction or failing regression test.
- Before/after expectations and the exact verification commands.
- A scoped contribution plan small enough to review in one sitting.

**Gate**

- The problem is current, non-duplicative, reproducible, and testable.

### December 14–20 — Submit the first upstream contribution

**OSS work**

- Implement the smallest complete fix or improvement.
- Add the regression test, benchmark, or documentation needed to prove it.
- Run focused tests and formatting locally; record results.
- Open a focused PR that explains the problem, approach, evidence, limitations,
  hardware/software environment, and any required AI-assistance disclosure.
- Respond to review comments with either a change or a reasoned explanation.

**Gate**

- A reviewer can reproduce the claim and understand the diff without relying
  on unstated context.

### December 21–31 — Deepen and consolidate

Choose based on the first contribution's state:

- If it is under review, prioritize fast, careful review responses and improve
  its evidence.
- If it merged, take a nearby follow-up in the same subsystem rather than
  jumping to an unrelated feature.
- If it was rejected for sound reasons, write down the lesson and apply it to a
  better-scoped task.
- If blocked on hardware or maintainer input, contribute a substantive test,
  benchmark, reproduction, or documentation improvement that still advances
  the same subsystem.

**Produce**

- A year-end report connecting the local GPT-2 experiments to the selected
  engine subsystem.
- A contribution ledger with issue/PR links, review status, tests, benchmarks,
  feedback received, and the next technical question.
- A Q1 2027 choice: deepen the same subsystem or return to the broader roadmap
  for Triton kernels, paged caching, scheduling, or distributed inference.

## Reading plan for *Inference Engineering*

This is intentionally selective rather than a race to finish the book.

| Date | Reading | Reason |
| --- | --- | --- |
| Completed by Oct 13 | Preface and Chapter 0, “Inference” | Establish the field and project vocabulary. |
| Oct 13–18 | Chapter 1, “Prerequisites” | Fill only the math, systems, and ML gaps needed by later chapters. |
| Oct 19–25 | Chapter 2 through section 2.2 | Connect model execution to prefill and decode. |
| Oct 26–Nov 1 | Sections 2.4–2.5 | Frame bottlenecks and optimization work. |
| Nov 9–15 | Revisit section 2.4 while building the cost model | Turn the prose into estimates and predictions. |
| Nov 16–22 | Chapter 3, “Hardware” | Build the GPU execution and memory model needed for profiling. |
| Nov 23–29 | Chapter 4, “Software” | Place frameworks, kernels, engines, and profiling in one stack. |
| Dec 1–20 | Relevant parts of Chapter 5 only | Read caching, batching, quantization, or speculative decoding in response to the chosen subsystem. |

For each reading block, record three things: one mechanism in my own words,
one prediction it makes about the current code, and one question to settle by
measurement or source inspection.

## Contribution rules

- Choose work from a current issue, roadmap item, measured bug, or demonstrated
  documentation/test gap—not from a desire to have a PR.
- Search current issues and open PRs immediately before starting.
- Talk to maintainers before a large feature, architectural change, or costly
  benchmark effort.
- Prefer a regression test plus a focused fix over an unverified refactor.
- Never claim a performance improvement without a reproducible baseline,
  representative workload, correctness check, and hardware/software details.
- Treat AI as a tutor and reviewer. I must understand and defend every upstream
  line, follow the target repository's AI policy, and disclose assistance where
  required.
- Never use upstream CI as the first attempt at basic debugging when a relevant
  local check is available.

## Progress log

Update this section weekly without rewriting the plan around temporary delays.

| Date | Evidence produced | Reading completed | Main unresolved question | Next gate |
| --- | --- | --- | --- | --- |
| 2026-10-13 | Session 1 architecture diagram and GPT-2 module skeleton | Preface + Chapter 0 of *Inference Engineering* | How do attention tensors and repeated generation map to inference cost? | Complete the attention/MLP shape trace. |

