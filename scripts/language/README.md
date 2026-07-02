# Language Embedding Ablations

This directory is for offline language-goal embedding tests. These scripts do
not launch Isaac or train a planner; they ask whether candidate text embedding
models make the motion-goal texts separable enough to serve as planner goals.

Run the fast local smoke test:

```bash
pixi run python scripts/language/ablate_language_embedding_models.py \
  --model_set smoke \
  --prompt_tiers attribute_text
```

Run the heavier local sentence-transformer set:

```bash
pixi run python scripts/language/ablate_language_embedding_models.py \
  --model_set recommended \
  --prompt_tiers attribute_text kinematic_description \
  --trust_remote_code
```

Run Ollama embedding models from a local Ollama server:

```bash
ollama serve
pixi run python scripts/language/ablate_language_embedding_models.py \
  --model_set ollama \
  --prompt_tiers attribute_text \
  --pull_ollama
```

For a probe that only records which Ollama models are missing, add
`--allow_zero_success`.

Useful custom model examples:

```bash
pixi run python scripts/language/ablate_language_embedding_models.py \
  --model_set smoke \
  --spec sentence-transformer:BAAI/bge-large-en-v1.5 \
  --spec ollama:nomic-embed-text \
  --prompt_tiers attribute_text
```

Each run writes:

- `tables/*.pt`: SkillCommander-compatible embedding tables.
- `audits/*.audit.json`: existing language-space diagnostics plus collision
  metrics.
- `language_embedding_ablation_summary.{json,csv,md}`: ranked report.

The most useful columns are nearest-neighbor category accuracy, intra/inter
cosine margin, best-same-vs-best-different gap rate, wrong top-1 count, and
cross-category near-duplicate count.

Rows are ranked by cosine separation by default: larger intra-minus-inter margin
first, then lower inter-category cosine. Use `--rank_by accuracy` or
`--rank_by balanced` for alternate orderings.

E5-family sentence-transformer models automatically embed `passage: <text>` by
default, matching their model-card usage pattern. Override with `--e5_prefix`
or disable with `--disable_model_prompt_adapters`.
