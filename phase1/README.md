# Phase 1: Research Direction and Handoff

Phase 1 turns an initial wireless research topic into a Phase-2-ready optimization problem handoff. The controller coordinates ScoutAgent and LiteratureAgent, stores their artifacts, applies gates, and exports the accepted problem direction for downstream modeling.

## Run

```bash
python3 phase1/scripts/run_phase1_pipeline.py --topic "cell-free massive MIMO"
```

Use `--run-id <id>` to choose a run folder name. If omitted, the top-level WARA runner creates a timestamped run id and passes it through all phases.

Phase 1 outputs are written under:

```text
outputs/paper_runs/phase1/<run_id>/
```

## Main Outputs

- `phase1_controller_manifest.json`: controller record of agent calls, gate results, frozen artifacts, and repair actions.
- `phase1-1/`: research frame for the wireless system, variables, objectives, constraints, and metrics.
- `phase1-2/`: literature grounding artifacts, references, abstract/PDF-backed literature cards, paper-reading records, gap signals, related models, baselines, and citation needs.
- `phase1-3/`: candidate research problems, selected problem, and selection rationale.
- `phase1-4/phase1_handoff.json`: frozen handoff consumed by Phase 2.

## Scope

Phase 1 is responsible for research framing, literature grounding, candidate-problem generation, candidate selection, and handoff creation. It does not design algorithms, generate experiment code, run simulations, or write the final manuscript.

The key output is `phase1_handoff.json`, which records the selected wireless optimization problem, modeling seed, expected objective and constraints, validation intent, reference support, abstract/PDF-grounded gap signals, and technical risks.

## Literature Grounding

Subphase 1.2 searches academic sources and builds literature cards only from records with readable content. WARA first uses abstracts from Semantic Scholar, OpenAlex, arXiv, IEEE Xplore, or other enabled sources. When a PDF URL is available, the controller may extract text from the first pages and record the extraction status. If PDF extraction fails, WARA falls back to the abstract. Metadata-only records may still be retained for the reference bank, but they are not used as gap-grounding cards unless metadata fallback is explicitly enabled for debugging.

## LLM Profiles

Use `--model-profile` to select a supported backend profile. This release supports Kimi, OpenAI, and DeepSeek profiles from the shared WARA-wide registry in `wara_core/llm/profiles.py`. Configure credentials locally through `.env` or shell environment variables; do not commit real API keys.
