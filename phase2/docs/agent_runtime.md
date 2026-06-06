# WARA Agent Runtime

WARA uses role agents organized by phase. The canonical backend package is
`wara_core/agents`, which exposes the named role-agent classes and the shared
agent registry. Each agent reads declared artifacts, writes declared artifacts,
and is coordinated by the controller through frozen contracts and bounded
repair paths.

## Role Agents

The active registry defines these role agents:

- `ScoutAgent`: frames the wireless research object and freezes the selected direction.
- `LiteratureAgent`: grounds references, evidence packs, and citation mapping.
- `FormulationAgent`: writes the system model, problem formulation, and mathematical contract.
- `TheoryAgent`: audits tractability, chooses the reformulation/solution route, and writes the algorithm contract.
- `ExperimentAgent`: generates the executable experiment package from frozen contracts.
- `ValidationAgent`: packages Phase2.4 outputs into reusable evidence for writing without regenerating experiments.
- `AnalysisAgent`: turns verified figures and metrics into numerical-results claims and insight.
- `WritingAgent`: assembles paper prose from frozen contracts and verified evidence.
- `ReviewAgent`: checks abbreviation, notation, evidence support, and paper readiness.
- `RepairAgent`: applies scoped fixes to one reported issue without reopening frozen contracts.

The `Controller` is not a content agent. It owns artifact registration, context assembly, contract freezing, gate records, retry limits, and failure routing.

Each run writes a lightweight `controller_manifest.json` index at the run root, plus separate controller ledgers for Phase 2 and Phase 3. The Phase 2 controller writes `phase2_controller_manifest.json`, which is the source of truth for modeling, formulation, algorithm, implementation, and evidence gates. The Phase 3 controller writes `phase3_controller_manifest.json`, which consumes the frozen Phase 2 handoff and records synthesis, citation, review, and repair gates. The handoff between them is materialized as `phase2-5/phase2_to_phase3_handoff.json`.

## Active Phase Mapping

- `phase1.1` -> `ScoutAgent`: research framing.
- `phase1.2` -> `LiteratureAgent`: evidence grounding.
- `phase1.3` -> `ScoutAgent`: direction contract.
- `phase1.4` -> `Controller`: WARA handoff materialization.
- `phase2.1` -> `FormulationAgent`: system model and problem formulation.
- `phase2.2` -> `TheoryAgent`: tractability route and reformulation path.
- `phase2.3` -> `TheoryAgent`: algorithm design and experiment blueprint.
- `phase3.1` -> `WritingAgent`: technical section drafting.
- `phase2.4` -> `ExperimentAgent`: executable experiment generation, validation, benchmark selection, and figures.
- `phase2.5` -> `ValidationAgent`: result verification and figure promotion from Phase 2.4 outputs.
- `phase3.1` -> `WritingAgent`: technical sections drafting from frozen contracts.
- `phase3.2` -> `AnalysisAgent`: numerical-results writing.
- `phase3.3` -> `WritingAgent`: technical section assembly.
- `phase3.4` -> `LiteratureAgent`: introduction and references.
- `phase3.5` -> `ReviewAgent`: final review and repair routing.
- `phase3.6` -> `RepairAgent`: final scoped revision.

## LLM Boundary

Phase2.4 is the main LLM-driven experiment implementation boundary. The request should include the frozen mathematical contract, algorithm contract, figure intent, KPI definitions, and benchmark requirements, then ask for one runnable experiment package.

Phase 2.5 must not redesign or regenerate experiments. It reads Phase 2.4 artifacts and emits the verified result package needed by Phase 3: figure metadata, benchmark definitions, reported KPIs, and supported claims.

## Contract Rule

Downstream agents may implement, analyze, cite, or repair artifacts, but they must not silently change the selected topic, mathematical contract, algorithm contract, benchmark identities, or verified experiment evidence. If one of those contracts is wrong, the controller must reopen the correct upstream phase explicitly.
