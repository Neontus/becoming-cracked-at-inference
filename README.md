# Becoming Cracked at Inference

A learning lab for understanding LLM inference systems from first principles.
The goal is not to collect tutorials or ship a wrapper around an existing
engine. The goal is to derive, implement, measure, profile, and explain the
systems that make autoregressive models run.

## Repository map

```text
src/inference_lab/ implementation code
tests/             correctness tests
posts/             public writing for neontus.github.io
scripts/           note and publishing helpers
ROADMAP.md         the learning sequence
```

Code for the inference engine belongs under `src/inference_lab/`. Add
subpackages such as `model`, `engine`, `kernels`, `scheduler`, and `kv_cache`
only when the curriculum reaches those problems. Their shape should emerge from
the implementation rather than being guessed in advance.

## Set up and run code

Create an isolated environment and install the repository in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable .
```

Confirm the package runs:

```bash
python -m inference_lab
```

Run the correctness tests:

```bash
python -m unittest discover --start-directory tests
```

As dependencies become necessary, add them deliberately to `pyproject.toml`
and document why they are needed. PyTorch, Triton, CUDA tooling, and profiling
dependencies are intentionally not installed before the curriculum reaches
them.

## Start a work session

Create a dated session note:

```bash
python3 scripts/new_session.py "naive decoding"
```

Spend five minutes at the end of the session recording what changed, what was
measured, what was surprising, and what remains unclear. These logs are source
material, not polished posts.

## Draft a public note

Create a draft from the public-note template:

```bash
python3 scripts/new_public_note.py kv-cache-from-first-principles \
  "KV Caching From First Principles"
```

Write the first technical explanation yourself. Set `draft: false` only after
the note contains an actual implementation or experiment and passes:

```bash
python3 scripts/check_public_notes.py
```

## Publish to the portfolio

The portfolio repository renders Markdown from `content/writing/` and already
deploys to GitHub Pages whenever `main` is pushed.

```bash
./scripts/publish_to_site.sh \
  posts/kv-cache-from-first-principles.md \
  /path/to/Neontus.github.io
```

Then review and commit both repositories:

```bash
git add posts ROADMAP.md
git commit -m "Document KV-cache experiment"

cd /path/to/Neontus.github.io
npm run build
git add content/writing
git commit -m "Publish KV-cache field note"
git push origin main
```

The publish helper refuses drafts and validates the destination before copying.
It does not commit or push anything on your behalf.

## Documentation rhythm

- Every serious session (5 minutes): add a raw session note.
- Every week or useful result (30–60 minutes): turn one insight into a public
  note with standalone value. Prefer “Why decode is memory-bound” over “Week 3.”
- At major milestones: publish a deeper report that connects design decisions,
  measurements, profiler evidence, and the corresponding code tag or commit.
- Every public claim about performance should include the hardware, software
  versions, workload, metric definition, and enough commands to reproduce it.

The detailed learning sequence lives in [ROADMAP.md](ROADMAP.md).
The collaboration rules for AI tools live in [AGENTS.md](AGENTS.md).
