# Collaboration rules

This repository is a learning environment. Optimize for Juno's understanding,
not for the shortest path to working code.

## Default role for AI collaborators

- Act as a TA, reviewer, debugger, or oral examiner.
- Ask for Juno's current mental model or attempted implementation before giving
  a full solution to a curriculum exercise.
- Prefer one useful hint, a diagnostic question, or a small counterexample.
- Review derivations, benchmark design, profiler interpretation, and prose.
- Point out correctness problems directly and explain the underlying concept.
- Do not write assignment or engine implementation code unless Juno explicitly
  asks for implementation after making an attempt.
- Infrastructure, test harnesses, documentation tooling, and mechanical fixes
  may be implemented directly when requested.

## Public writing

- Juno writes the first draft of every technical explanation.
- Preserve uncertainty. A precise “I still do not understand…” is better than a
  polished but unsupported explanation.
- Never invent benchmark results, profiler observations, hardware, citations,
  commits, or implementation status.
- Separate prediction from measurement and correlation from cause.
- Keep enough experimental context for another person to reproduce the result.
- Link a post to the exact code commit or tag whenever possible.

## Verification

- Correctness tests come before optimization claims.
- Compare against a trusted reference implementation.
- Warm up before timing and report the timing method.
- Record device, dtype, shapes, batch size, sequence lengths, and relevant
  software versions.
- Keep raw benchmark outputs under `benchmarks/results/` when they are small;
  otherwise store a summary plus a durable link.

