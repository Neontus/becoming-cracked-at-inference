# Experiments

Each non-trivial experiment should record:

- the question and pre-measurement prediction;
- the code commit;
- hardware and software environment;
- model, dtype, tensor shapes, sequence lengths, and batching/request pattern;
- correctness criteria and reference implementation;
- warmup, repetitions, synchronization, and reported statistics;
- raw result location and a short interpretation.

Prefer a small configuration file plus a reproducible command over an informal
description that cannot be rerun.

