# WARA

[English](README.md) | [中文](README.zh-CN.md)

WARA, the Wireless AutoResearch Agent, is a closed-loop research automation workspace for wireless optimization papers. It turns a broad wireless topic into a traceable research package: literature-grounded problem proposal, optimization model, solution route, executable experiments, verified evidence, and a reviewed paper package.

![WARA console overview](docs/assets/wara_console_overview.png)

## Why WARA

Wireless optimization research is not a single writing task. A credible paper needs a coherent chain from research gap, system model, mathematical formulation, algorithm design, executable experiments, numerical evidence, and final manuscript claims. WARA treats that chain as controller-managed artifacts instead of one-shot text generation.

Core ideas:

- **Three-phase research workflow:** problem proposal, technical study, and paper-package generation.
- **Artifact-mediated control:** each phase reads declared inputs and writes explicit artifacts for downstream reuse.
- **Frozen contracts:** selected problem, mathematical formulation, algorithm route, and verified evidence are frozen before later stages consume them.
- **Localized repair:** failed gates are routed back to the responsible artifact owner instead of restarting the whole run.
- **Independent run folders:** every run writes to its own output directory for reproducibility and inspection.

## Manuscript Review Agent

The web console also includes an independent **Manuscript Review** page aligned with the paper evaluation setup. It accepts a manuscript PDF, sends only the extracted PDF text to the selected scoring model, and returns two rubric-based scores: manuscript-level research validity and optimization research maturity. This scorer is separate from the three-phase WARA pipeline and does not read hidden run artifacts, TeX sources, figures, or manifests.

## Workflow

WARA organizes each run into three phases:

| Phase | Purpose | Main artifacts |
| --- | --- | --- |
| Phase 1 | Research gap identification and problem proposal | Research frame, literature evidence pack, candidate problems, frozen problem handoff |
| Phase 2 | Wireless optimization modeling, algorithm design, and experimentation | Mathematical contract, algorithm contract, executable experiment package, verified result package |
| Phase 3 | Research deliverable generation | Technical sections, full manuscript, review report, repaired final paper package |

Phase numbering follows the paper workflow:

- Phase 1 uses subphases `phase1.1` to `phase1.4`.
- Phase 2 uses subphases `phase2.1` to `phase2.5`.
- Phase 3 uses subphases `phase3.1` to `phase3.6`.

See [docs/phase_architecture.md](docs/phase_architecture.md) for the active phase mapping.

## Quick Start

Start the local web console:

```bash
python3 scripts/start_wara_ui.py --host 127.0.0.1 --port 8765
```

The startup script creates `.venv` when needed, installs `requirements.txt`, and starts the WARA console. Open:

```text
http://127.0.0.1:8765
```

To force reinstall runtime requirements:

```bash
python3 scripts/start_wara_ui.py --install --host 127.0.0.1 --port 8765
```

Command-line runs are also supported after setup:

```bash
python3 scripts/start_wara_ui.py --install --host 127.0.0.1 --port 8765
.venv/bin/python run_wara_pipeline.py --help
.venv/bin/python run_wara_pipeline.py --topic "cell-free massive MIMO" --run-id wara001
```

If `--run-id` is omitted, the top-level launcher creates a timestamped run id and passes it to all three phases.

## Model Providers

WARA supports a single selected model profile at a time. The web console exposes one API-key input for the currently selected model provider.

Supported providers in this release:

- Kimi / Moonshot
- OpenAI
- DeepSeek

Create a local `.env` file from the template:

```bash
cp .env.example .env
```

Then fill only the keys you need:

```text
KIMI_API_KEY=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
```

The release package intentionally does not include real API keys.

## Output Layout

Each run writes to an independent folder under `outputs/paper_runs/`:

```text
outputs/paper_runs/
  phase1/<run_id>/
  phase2/<run_id>/
  phase3/<run_id>/
  final_papers/<run_id>/
```

This keeps Phase 1, Phase 2, Phase 3, and final-paper artifacts aligned under the same `<run_id>`.

## Repository Structure

```text
phase1/                 Research direction discovery and literature grounding
phase2/                 Modeling, algorithm route, experiments, and evidence validation
phase3/                 Writing, citation integration, review, repair, and export
wara_core/              Shared agents, LLM profiles, literature tools, and packaging utilities
app/                    Local WARA web console
evaluation/             Independent manuscript review agent
scripts/                Console launcher and backend server
docs/                   Architecture notes and showcase assets
outputs/                Ignored runtime output root
```
