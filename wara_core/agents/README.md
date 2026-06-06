# WARA Agents

This package is the canonical backend interface for WARA role-specialized agents.

The runtime uses ten role agents:

- ScoutAgent
- LiteratureAgent
- FormulationAgent
- TheoryAgent
- ExperimentAgent
- ValidationAgent
- AnalysisAgent
- WritingAgent
- ReviewAgent
- RepairAgent

Most roles share the generic artifact-mediated `RoleAgent` runtime. `ExperimentAgent`
has a specialized runtime because it owns executable experiment code, validation
harness interaction, result tables, solver logs, and figure artifacts.

The older `phase2/agents` package is kept as a compatibility layer for existing
phase scripts and tests.
