# CadQuery knowledge probe

This probe asks a model about each API file in a local CadQuery documentation
snapshot. Only the exact API name and a fixed question reach the model. The
documentation body is reserved for assessment. Each request uses a fresh
conversation, no tools, temperature zero, and the served model's chat template.

Place the snapshot in `resources_servers/deepcad/cadquery-docs/` (ignored by Git).
The HSG setup uses the project's `rem getnode` allocation and the container's
vLLM worker Python environment. `serve.py` applies the same source compatibility
patches as NeMo-RL before starting the standard vLLM HTTP server. Pass ordinary
vLLM arguments, including `--model`, `--chat-template`, and `--tensor-parallel-size`.

From the Gym root, collect responses:

```bash
python -m resources_servers.deepcad.knowledge_eval.probe \
  --docs resources_servers/deepcad/cadquery-docs \
  --output resources_servers/deepcad/knowledge-results/step30 \
  --base-url http://localhost:8000/v1 \
  --model cadquery-knowledge-step30 --concurrency 16 --max-tokens 8192
```

Each API gets a Markdown file with the prompt and verbatim response, and a JSON
file with the exact request, response, token usage, and finish reason. Reasoning
is stored separately from the final answer. Truncated requests retry with a
larger generation budget, up to 28,000 tokens. Failures are recorded in
`collection.json`; rerunning resumes successful responses. A manifest identifies
every source file by its SHA-256 hash. Use a separate output for a different model
or protocol.

Produce the static signature comparison:

```bash
python -m resources_servers.deepcad.knowledge_eval.signatures \
  --docs resources_servers/deepcad/cadquery-docs \
  --responses resources_servers/deepcad/knowledge-results/step30
```

The comparison checks explicit parameter names/order, parameter kinds, and default
expressions against parseable documented signatures. It ignores annotations and
return types. Unparseable reference signatures and noncallable entries have an
explicit exclusion status. A match to one overload is enough; this does not
establish complete overload recall. Equivalent defaults written differently can
fail the strict expression check. Model answers in unsupported signature formats
can also fail parsing. Review the per-item comparison JSON and Markdown answers
before interpreting these counts.

Purpose, behavioral correctness, hallucinations, and example validity require
separate review against the references. Signature recall alone does not measure
working CAD generation. The snapshot includes aliases and Python housekeeping
members, so counts over all API paths should be reported alongside the modeling
API groups. Record the documentation version; a development snapshot can contain
APIs newer than the model's training data.
