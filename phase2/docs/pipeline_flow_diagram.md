# WARA Backend Pipeline Diagram

This diagram documents the backend-only WARA flow. Runs are launched from CLI entry points and managed by the phase controllers.

```mermaid
flowchart TD
    U["User topic or existing Phase 1 run"] --> TOP["Top-level backend runner<br/>run_wara_pipeline.py"]
    TOP --> P1["Phase 1 runner<br/>phase1/scripts/run_phase1_pipeline.py"]
    P1 --> H["Phase 1 handoff<br/>phase1_handoff.json"]
    H --> P2["Phase 2 runner<br/>phase2/scripts/run_phase2_pipeline.py"]
    P2 --> H23["Phase 2 to Phase 3 handoff<br/>phase2_to_phase3_handoff.json"]
    H23 --> P3["Phase 3 runner<br/>phase3/scripts/run_phase3_pipeline.py"]

    subgraph PHASE1["Phase 1: Research Direction"]
        P11["phase1.1<br/>Research Framing"]
        P12["phase1.2<br/>Evidence Grounding"]
        P13["phase1.3<br/>Direction Contract"]
        P14["phase1.4<br/>WARA Handoff"]
        P11 --> P12 --> P13 --> P14
    end

    subgraph PHASE2["Phase 2: Model, Algorithm, Experiment"]
        P21["phase2.1<br/>Construct System Model and Problem"]
        P22["phase2.2<br/>Check Convexity and Reformulation Route"]
        P23["phase2.3<br/>Specify Algorithm and Experiment Plan"]
        P24["phase2.4<br/>Implement and Run Experiment Code"]
        P25["phase2.5<br/>Verify Results and Promote Figures"]
        P21 --> P22 --> P23 --> P24 --> P25
    end

    subgraph PHASE3["Phase 3: Writing, Review, Revision"]
        P31["phase3.1<br/>Technical Sections Drafting"]
        P32["phase3.2<br/>Numerical Results Writing"]
        P33["phase3.3<br/>Technical Sections Assembly"]
        P34["phase3.4<br/>Introduction & Reference Curation"]
        P35["phase3.5<br/>Final Review / Pre-submission Review"]
        P36["phase3.6<br/>Final Revision / Apply Review Fixes"]
        P31 --> P32 --> P33 --> P34 --> P35 --> P36
    end

    P14 --> H
    P2 --> P21
    P25 --> H23
    P3 --> P31
    P35 --> OUT["Final artifacts<br/>revised_full_paper.tex<br/>revised_full_paper_preview.pdf<br/>review reports"]
```

## Artifact Boundary

- Phase 1 writes research-direction contracts and the handoff consumed by Phase 2.
- Phase 2 writes the mathematical, algorithmic, validation, and evidence artifacts.
- Phase 3 writes the paper-facing artifacts, review reports, and final revision.
- Runtime outputs are kept under phase run directories and ignored by git.
