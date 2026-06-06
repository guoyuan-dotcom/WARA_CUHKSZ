from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


PHASE1_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PHASE1_ROOT.parent
ENGINE_ROOT = PHASE1_ROOT / "engine"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from wara_core.llm.client import LLMClient, LLMConfig  # noqa: E402
from wara_core.llm.profiles import (  # noqa: E402
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILES,
    get_model_profile,
    normalize_model_profile,
)
from wara_core.domains.wireless_ontology import extract_wireless_ontology, looks_like_wireless_topic  # noqa: E402
from wara_core.domains.wireless_topic_taxonomy import (  # noqa: E402
    assess_wireless_topic_taxonomy_candidate,
    build_wireless_topic_taxonomy_plan,
)
from wara_core.data import load_seminal_papers  # noqa: E402


DEFAULT_MAX_TOKENS = 24000

PHASE_NAMES = {
    1: "Research Framing",
    2: "Evidence Grounding",
    3: "Direction Contract",
    4: "WARA Handoff",
}
PHASE_IDS = {
    1: "phase1.1",
    2: "phase1.2",
    3: "phase1.3",
    4: "phase1.4",
}


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Any:
        ...


@dataclass(frozen=True)
class WaraPhase1Result:
    run_id: str
    run_dir: Path
    handoff_dir: Path
    handoff_file: Path
    selected_title: str


def make_run_id(topic: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:6]
    return f"wara-phase1-{stamp}-{suffix}"


def run_wara_phase1(
    *,
    topic: str,
    run_dir: Path,
    model_profile: str = DEFAULT_MODEL_PROFILE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tail_root: Path | None = None,
    llm_client: ChatClient | None = None,
) -> WaraPhase1Result:
    try:
        from phase1_controller import Phase1Controller
    except ModuleNotFoundError:  # pragma: no cover - package import path fallback
        from phase1.scripts.phase1_controller import Phase1Controller

    return Phase1Controller(
        topic=topic,
        run_dir=run_dir,
        model_profile=model_profile,
        max_tokens=max_tokens,
        tail_root=tail_root,
        llm_client=llm_client,
    ).run()


def make_llm_client(model_profile: str = DEFAULT_MODEL_PROFILE) -> LLMClient:
    _load_workspace_env()
    model_profile = normalize_model_profile(model_profile)
    profile = get_model_profile(model_profile)
    api_key_env = profile["api_key_env"]
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    fallback_models: list[str] = []
    return LLMClient(
        LLMConfig(
            base_url=(
                os.environ.get("KIMI_BASE_URL") or os.environ.get("MOONSHOT_BASE_URL")
                if model_profile.startswith("kimi-")
                else os.environ.get("DEEPSEEK_BASE_URL")
                if model_profile.startswith("deepseek-")
                else os.environ.get("OPENAI_BASE_URL")
            )
            or profile["base_url"],
            api_key=api_key,
            wire_api=profile["wire_api"],
            primary_model=profile["primary_model"],
            fallback_models=fallback_models,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.4,
            timeout_sec=int(os.environ.get("WARA_LLM_TIMEOUT_SEC", "900") or 900),
            max_retries=int(os.environ.get("WARA_LLM_MAX_RETRIES", "3") or 3),
            retry_base_delay=float(os.environ.get("WARA_LLM_RETRY_BASE_DELAY", "8") or 8),
            retry_max_delay=float(os.environ.get("WARA_LLM_RETRY_MAX_DELAY", "60") or 60),
        )
    )


def build_research_object_prompt(topic: str, scope_context: dict[str, Any]) -> tuple[str, str]:
    scope_contract = dict(scope_context.get("scope_contract") or {})
    forbidden = format_inline_list(scope_contract.get("forbidden_added_mechanisms"))
    extension_axes = scope_contract.get("candidate_extension_axes") or []
    concrete_extensions = format_inline_list(scope_contract.get("candidate_extension_mechanisms"))
    extension_policy = dict(scope_contract.get("mechanism_extension_policy") or {})
    system = (
        "You are WARA Phase-1 ResearchObjectAgent. "
        "Your job is to turn a wireless user topic into a precise research object. "
        "Do not write a paper, do not choose empirical comparison methods, and do not redesign the topic. "
        "Freeze only the research semantics that downstream formulation agents need. "
        "Quality must be built at the source: do not leave vague novelty, vague KPIs, or vague modeling choices for downstream review to discover."
    )
    user = f"""
Define the WARA Phase-1 research object for the user topic below.

User topic:
{topic}

Wireless scope artifact:
{json.dumps(scope_context, ensure_ascii=False, indent=2)}

Return JSON only with exactly these top-level keys:
topic_profile, research_object, wireless_system_seed, mechanism_hypothesis, phase2_readiness, direction_constraints

Schema requirements:
- topic_profile: object with domain, user_topic, preserved_mechanisms, candidate_extension_axes, candidate_extension_mechanisms, forbidden_added_mechanisms, scope_boundary, phase2_risks.
- research_object: object with research_question, physical_mechanism, decision_layer, performance_gap, expected_research_gain, non_goals.
- wireless_system_seed: object with nodes, channel_model_seed, csi_assumption_seed, controls, parameters, derived_quantities, primary_kpis, constraints_seed.
- mechanism_hypothesis: object with why_gain_may_exist, operating_regimes, failure_regimes, evidence_needed.
- phase2_readiness: object with formulation_needs, theory_needs, validation_needs, ambiguity_to_resolve.
- direction_constraints: object with must_preserve, must_not_add, allowed_abstractions.

Design rules:
- First-pass research-quality contract: the first research object should already be Phase-2-ready in spirit. It must name the physical resource/coupling, the controllable decision layer, the expected KPI effect, the likely operating regime, and the assumptions that must be formalized later.
- Do not defer core research choices to review or later repair. If a mechanism, KPI, or assumption is too vague to support formulation, mark it as ambiguity_to_resolve instead of pretending it is a contribution.
- Preserve the user's requested technology and scenario. Do not add mechanisms outside the wireless scope artifact.
- Scope-specific mechanisms that must not be added: {forbidden or "none beyond the topic boundary"}.
- Candidate extension axes, if any: {json.dumps(extension_axes, ensure_ascii=False, indent=2)}
- Preselected concrete extension mechanisms, if any: {concrete_extensions or "none; if an extension is useful, the LLM must select and justify the concrete mechanism from the high-level axes."}
- Extension policy: {json.dumps(extension_policy, ensure_ascii=False)}.
- The wireless taxonomy artifact may include concrete examples. Treat them as a coverage guide across wireless levels, not as defaults, limits, or recommendations.
- If extension policy is enabled, define the plain in-scope research object first. Do not bake an added mechanism into the base research object; concrete extension mechanisms belong in Phase 1.3 candidate directions after evidence/gap reasoning.
- Do not decide the winning novelty mechanism in the research object. This phase should identify the base coupling, open missing layers, and possible evidence needs without steering the next agent toward a familiar mechanism.
- Treat model-fidelity choices as modeling choices unless the user topic or verified evidence makes them the research gap. For powering topics, do not make the energy-conversion model itself the default ambiguity or novelty by itself.
- Keep scenario, resource granularity, uncertainty/reliability, propagation geometry, distributed coordination, and device/energy response equally open until the direction agent compares them.
- Treat the final paper as downstream expression; here we are defining the research object that Phase 2 can formalize.
- Keep the object compact enough for a short wireless letter, but prioritize research clarity over writing style.
- Use topic-grounded paper-style system KPIs. Choose metrics from the selected service/task and evidence rather than copying examples from the taxonomy.
- If the user topic does not specify a primary objective, do not silently default to minimizing BS/transmit power. Treat transmit power as a resource cost or diagnostic unless the research object is explicitly about required-resource minimization, energy saving, coverage under power limits, battery operation, or energy efficiency.
- For broad wireless optimization topics, first consider paper-facing service/performance objectives such as weighted system utility, sum rate/spectral efficiency, worst-user rate, reliability/outage, sensing accuracy/CRB/beampattern quality, harvested energy/DC power, latency, energy efficiency, service-region size, or a rate-energy-sensing tradeoff under a fixed resource budget.
- If several objective orientations are plausible, record the ambiguity in phase2_readiness.ambiguity_to_resolve or direction_constraints instead of collapsing the problem to minimum transmit power.
- Do not choose, name, or imply empirical comparison methods in Phase 1. Phase 2.4 owns empirical comparison design after the mathematical contract is frozen.
- Do not invent references or cite papers.
- Use ASCII math notation in JSON values.
""".strip()
    return system, user


def build_direction_contract_prompt(
    topic: str,
    research_frame: dict[str, Any],
    evidence_pack: dict[str, Any] | None = None,
) -> tuple[str, str]:
    evidence_pack = evidence_pack or {}
    scope_contract = dict(dict(research_frame.get("wireless_scope") or {}).get("scope_contract") or {})
    forbidden = format_inline_list(scope_contract.get("forbidden_added_mechanisms"))
    extension_axes = scope_contract.get("candidate_extension_axes") or []
    concrete_extensions = format_inline_list(scope_contract.get("candidate_extension_mechanisms"))
    extension_policy = dict(scope_contract.get("mechanism_extension_policy") or {})
    system = (
        "You are WARA Phase-1 ScoutAgent acting as the direction-contract specialist. "
        "Your job is to choose a Phase 2-ready wireless research direction and freeze the Phase 2 handoff contract in one pass. "
        "Avoid rewriting the same problem multiple times: keep alternatives lightweight, but make the selected contract precise. "
        "Quality must be built during direction selection, not delegated to downstream review."
    )
    user = f"""
Create the WARA Phase-1 direction decision and Phase-2 handoff contract.

User topic:
{topic}

Research framing artifact:
{json.dumps(research_frame, ensure_ascii=False, indent=2)}

Evidence grounding artifact:
{json.dumps(evidence_pack, ensure_ascii=False, indent=2)}

Return JSON only with exactly these top-level keys:
candidate_directions, selection_decision, selected_candidate, problem_contract_seed, novelty_contract, proof_contract, validation_contract, kill_criteria, handoff_notes

Before choosing candidates, reason internally using this constructive template:
1. Start from the base wireless resource in the research frame.
2. Identify the performance bottleneck or research gap before choosing a mechanism. A gap may be a missing operating regime, an underspecified resource-control layer, an unexplored coupling, a hard-to-solve feasible set, a robustness/reliability issue, a deployment/geometry variable, or a model-response effect.
3. Inspect the high-level extension axes only after the gap is stated. For each open axis, decide whether it actually repairs that gap through a change in physical marginal value, spatial coupling, feasibility boundary, uncertainty margin, deployment geometry, coordination structure, hardware response, or resource granularity.
4. The controller does not preselect technologies. If an extension is useful, choose the concrete mechanism yourself and justify why that mechanism follows from the gap, topic, and evidence rather than from familiarity.
5. Keep only candidates where the selected mechanism or plain formulation creates a joint-optimization gain, not just a broader title.
6. Prefer candidates whose gain can be shown by ordinary paper-level KPIs and operating-regime plots, but do not choose the easiest-to-plot mechanism if another candidate has a stronger research gap.
7. Treat technology-combination novelty as valid only when the combination creates a new coupling, tradeoff, constraint structure, algorithmic difficulty, or operating regime that is absent from either component alone.
8. Since this project is optimization-centered, ask how the gap changes the objective, constraints, feasible set, surrogate design, decomposition, or algorithmic route.
9. Translate the best gap-to-mechanism story into the required JSON fields; do not include this hidden reasoning as prose outside JSON.

Candidate requirements:
- candidate_directions must contain 2 or 3 lightweight alternatives.
- Candidate titles and selected_candidate.title are Phase-2 working direction titles. They must be IEEE-style and title case, but they may remain technical because downstream modeling and experiments consume them.
- selected_candidate.paper_title is the paper-facing manuscript title used only for final writing. It should be shorter and more natural than the working direction title while preserving the same research object.
- paper_title must avoid template-like objective/method/scenario stacking such as "Weighted Sum-Rate X for Y" when a more natural title is available. Prefer a title that names the wireless mechanism and setting, e.g., "Power-Constrained RIS Beamforming for Integrated Sensing and Communication" or "Near-Field Beamfocusing for Multiuser Downlink Communications".
- Each alternative must contain id, title, problem_statement, wireless_scenario, research_angle, selected_extension_axis, concrete_mechanism, mechanism_for_gain, mechanism_interaction, resource_coupling_change, expected_kpi_gain, operating_regime, tractability_risk, combination_novelty, why_not_keyword_stacking, new_coupling_or_tradeoff, performance_bottleneck_addressed, testable_gain_regime, optimization_gap, optimization_novelty, objective_constraint_structure, algorithmic_route, evidence_alignment, phase2_risks, kill_criteria.
- Alternatives should help decide the direction; they should not each restate a full mathematical contract.
- If mechanism_extension_policy.enabled is true, include one plain in-scope candidate and at most two one-mechanism extension candidates. For the plain candidate, set selected_extension_axis="none" and concrete_mechanism="none". For extension candidates, pick concrete mechanisms yourself from the high-level extension axes; do not assume the controller has named the right technology.
- If mechanism_extension_policy.enabled is false, all candidates must stay inside the preserved mechanisms.
- Any extension candidate must name selected_extension_axis and concrete_mechanism, then explain why that mechanism changes the base resource coupling and why it is not a topic-stacking gimmick.
- Any extension candidate with empty or generic mechanism_interaction, resource_coupling_change, expected_kpi_gain, or operating_regime is invalid.
- The plain candidate is a serious candidate, not a straw man. Plain and extension candidates must be evaluated by the same rubric: gap specificity, optimization contribution, physical coupling, evidence plausibility, Phase 2 implementability, and experimental observability.
- Do not select an extension merely because the plain candidate sounds generic. Select an extension only if it repairs a sharper gap than the plain formulation and can be formulated without uncontrolled complexity.

Selected-contract requirements:
- selected_candidate must contain title, paper_title, problem_statement, wireless_scenario, objective, variables, core_constraints, claimed_contribution, novelty_delta, optimization_structure, source_of_difficulty, tractability_path, theorem_or_algorithmic_claim, expected_research_gain.
- selected_candidate.title and selection_decision.selected_title must use title case and must be the same Phase-2 working direction title unless the decision field intentionally gives a shorter display title.
- selected_candidate.paper_title may be shorter than selected_candidate.title, but it must not change the selected mechanism, objective orientation, wireless scenario, or claim scope.
- Backward-compatible aliases source_of_nonconvexity and convexification_path may be included only when they are accurate. Do not force a nonconvex story; if the formulation is convex after a clean transformation, say so in source_of_difficulty/tractability_path and make the novelty rest on the wireless coupling, operating regime, or algorithmic/evidence value.
- problem_contract_seed must contain controls, parameters, derived_quantities, objective, constraints, assumptions, primary_kpis.
- novelty_contract must contain claim_boundary, prior_art_boundary, novelty_hypothesis, optimization_novelty, objective_constraint_delta, algorithmic_delta, main_risk.
- proof_contract must contain target_claims, assumptions, route, algorithmic_route, allowed_approximations.
- validation_contract must contain metrics, figures, parameter_sweeps, expected_trends, evidence_questions.
- selection_decision must contain selected_id, selected_title, rationale, rejected_ids, readiness_score_1_to_10.
- kill_criteria must be concrete reasons to stop or revise before Phase 2.
- handoff_notes must contain phase2_instructions and interface_warnings.

Design rules:
- First-pass direction-quality contract: choose a direction only if the gap, mechanism, formulation variables, proof/algorithm route, and evidence plan form a coherent paper story before review. The selected contract should already explain why the proposed optimization can plausibly beat a fair conventional design under a named operating regime.
- Do not rely on later phases to invent the contribution, primary KPI, tractability route, or experiment logic. If those elements cannot be stated constructively from the current evidence and topic, lower readiness_score_1_to_10 and put the uncertainty in kill_criteria or handoff_notes.
- Objective orientation is a deliberate research decision. For broad topics, compare plausible orientations when appropriate: maximize system utility/performance under resource budgets, maximize service fairness/reliability, or minimize required resource under fixed service constraints.
- Do not select transmit-power minimization simply because it is easy to solve, easy to plot, or common in QoS feasibility examples.
- If selected_candidate.objective minimizes transmit power or total resource, selected_candidate.problem_statement, claimed_contribution, validation_contract.metrics, and expected_research_gain must frame it as "less required resource to satisfy the same scoped service/reliability model." Otherwise choose a paper-facing service/performance objective or KPI from the topic.
- problem_contract_seed.primary_kpis and validation_contract.metrics must include the objective KPI and the active service KPIs that make the result meaningful. Transmit power must not be the only primary KPI unless required-resource minimization is explicitly the selected contribution.
- Preserve the user's requested technology and scenario; do not add mechanisms outside the research framing artifact.
	- Scope-specific mechanisms that must not be added: {forbidden or "none beyond the topic boundary"}.
	- Candidate extension axes, if any: {json.dumps(extension_axes, ensure_ascii=False, indent=2)}
	- Preselected concrete extension mechanisms, if any: {concrete_extensions or "none; choose concrete mechanisms only if they are justified by an axis, evidence, and optimization gain."}
	- Mechanism extension policy: {json.dumps(extension_policy, ensure_ascii=False)}.
	- Use the evidence grounding artifact to bound novelty and risk, but do not fabricate citations or priority claims. If the evidence artifact contains only query plans or empty retrieved references, treat mechanism-specific novelty as provisional and do not overclaim that one mechanism is literature-grounded.
	- If the evidence grounding artifact contains literature_cards or gap_signals, use them as grounding hints for candidate generation. A good candidate should connect its optimization_gap and evidence_alignment to specific card IDs, signal IDs, or explicit evidence needs.
	- Do not infer novelty from absence alone. If a relevant mechanism appears weakly covered because references are metadata-only or missing abstracts, state the gap as a provisional modeling or validation need rather than a literature priority claim.
	- Select the final research gap only when it is consistent with the research frame, at least one retrieved literature signal or evidence need, and a Phase-2-feasible optimization formulation.
	- Phase 1 must not choose empirical comparison methods. Phase 2.4 owns empirical comparisons after the mathematical contract is frozen.
- Make the research gain plausible through operating regimes, KPIs, and mechanism-level evidence needs, not by asking Phase 2 to tune blindly.
- Selection rubric priority: concrete research gap first, optimization contribution second, physical mechanism third, evidence plausibility fourth, Phase 2 implementability fifth, and experiment observability sixth.
- Positive mechanism-design goal: discover a mechanism interaction that lets joint optimization exploit coupling that separated/plain designs cannot exploit.
- Good mechanism stories usually have this shape: "because mechanism A changes the marginal value, geometry, uncertainty, or resource structure of resource B, the joint design reallocates the relevant controllable variables in regime C, improving KPI D."
- Use this research logic for mechanism combinations: mechanism interaction -> resource coupling change -> joint optimization gain -> operating-regime insight.
- Use this novelty logic for technology combinations: combination novelty is valid when the combination creates a new resource coupling, tradeoff, constraint structure, algorithmic route, or operating regime that is absent from either component alone.
- A strong candidate should identify the performance bottleneck in existing/plain designs, explain how the added mechanism repairs that bottleneck, and name the KPI/regime where the gain should appear.
- Optimization-centric rule: the final novelty must be expressible as a change in optimization formulation or solution route, such as a new or better-specified objective/constraint term, feasible-set geometry, surrogate construction, block update, decomposition, tractable approximation, or operating-regime characterization.
- Optimization difficulty may be nonconvexity, nonsmoothness, fractional structure, mixed discrete/continuous controls, uncertainty, coupled resources, saturation/threshold effects, scalability, or a convex-but-novel formulation with meaningful wireless insight. Nonconvexity can be a contribution only when it arises naturally from the wireless mechanism and the selected direction gives a credible solution route, proof scope, or empirical-validation route. Never invent nonconvexity just to make the paper look harder.
- Do not let the paper become a system-description paper. The selected direction must give Phase 2 enough information to formulate controls, derived quantities, objective, constraints, algorithm, and KPI sweeps.
- Reject title-only combinations. The added mechanism must change marginal value, feasibility, uncertainty, mobility, spatial coupling, or another physical resource-coupling property of the base wireless problem.
- Prefer a tractable framework, algorithmic route, operating-regime insight, and paper-level numerical evidence over hard theorem dependence.
- Do not make a specific relaxation, recovery property, optimality condition, or global optimality the required main novelty unless the evidence and assumptions make that claim low risk.
- For broad topics, do not rely on a fixed technology menu. Use the high-level axes to decide whether the strongest gap is in energy/hardware response, propagation geometry, mobility/deployment, uncertainty, distributed coordination, access/resource granularity, or another topic-grounded mechanism.
- For topics involving wireless powering, do not choose a device-response or model-fidelity mechanism by default. Choose one only when the gap analysis shows that response-shape behavior is the bottleneck and it beats alternatives such as geometry, resource granularity, reliability, topology, or plain tri-functional resource coupling.
- Selecting a plain formulation requires a concrete non-generic gap. Selecting an extension requires a concrete mechanism whose coupling can be formulated and simulated in Phase 2 without uncontrolled complexity.
- Validation metrics must be paper-style wireless system KPIs selected from the topic and contract. They should cover the active services and avoid abstract feasibility-only metrics unless the topic is explicitly feasibility-first.
- Do not use feasibility rate as a primary metric unless the topic is explicitly feasibility-first.
- Do not put unqualified "guarantee", "globally optimal", "exact", or "tight" claims in titles.
- Do not write experiment code. Do not write paper prose. Produce contracts only.
- Use ASCII math notation in JSON values.
""".strip()
    return system, user


def call_json_agent(
    client: ChatClient,
    *,
    agent_id: str,
    system: str,
    user: str,
    max_tokens: int,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    repair_rounds = max(0, int(os.environ.get("WARA_PHASE1_JSON_REPAIR_ROUNDS", "3")))
    current_user = user
    raw_content = ""
    errors: list[str] = []
    for attempt in range(repair_rounds + 1):
        response = client.chat(
            [{"role": "user", "content": current_user}],
            system=system,
            max_tokens=max_tokens,
            temperature=0.35,
            json_mode=True,
        )
        raw_content = str(getattr(response, "content", response) or "")
        try:
            payload = extract_json_object(raw_content)
            trace.append(
                {
                    "agent_id": agent_id,
                    "attempt": attempt + 1,
                    "status": "ok",
                    "prompt_chars": len(current_user),
                    "response_chars": len(raw_content),
                    "model": getattr(response, "model", ""),
                    "total_tokens": int(getattr(response, "total_tokens", 0) or 0),
                }
            )
            return payload
        except ValueError as exc:
            errors.append(str(exc))
            current_user = (
                "Repair the previous response into valid JSON only. "
                "Do not add markdown or explanation. "
                f"JSON parse error: {exc}\n\nPrevious response:\n{raw_content}"
            )
    trace.append(
        {
            "agent_id": agent_id,
            "status": "failed",
            "errors": errors,
            "response_chars": len(raw_content),
        }
    )
    raise ValueError(f"{agent_id} failed to return valid JSON: {'; '.join(errors)}")


def build_wireless_scope_context(topic: str) -> dict[str, Any]:
    ontology = extract_wireless_ontology(topic)
    taxonomy_plan = build_wireless_topic_taxonomy_plan(topic, max_blueprints=4)
    taxonomy_profile = dict(taxonomy_plan.get("input_profile") or {})
    taxonomy_layers = taxonomy_profile.get("layers", {}) if isinstance(taxonomy_profile, dict) else {}
    taxonomy_tags = {
        str(tag)
        for tags in taxonomy_layers.values()
        if isinstance(tags, list)
        for tag in tags
    }
    taxonomy_labels = [
        str(label)
        for layer in taxonomy_profile.get("display_layers", [])
        if isinstance(layer, dict)
        for label in coerce_list(layer.get("tag_labels"))
    ]
    ontology_tags = set(coerce_list(ontology.get("primary_tags")))
    is_wireless = bool(ontology.get("is_wireless")) or bool(taxonomy_tags)
    preserved_labels = dedupe_texts(coerce_list(ontology.get("primary_tag_labels")) + taxonomy_labels)
    preserved_tags = sorted(ontology_tags | taxonomy_tags)
    extension_policy = build_mechanism_extension_policy(topic, preserved_tags)
    scope_contract = {
        "domain": "wireless communications" if is_wireless else "wireless communications candidate",
        "user_topic": topic,
        "is_wireless_topic": is_wireless or looks_like_wireless_topic(topic),
        "preserved_mechanisms": preserved_labels,
        "preserved_ontology_tags": preserved_tags,
        "candidate_extension_axes": extension_policy.get("candidate_extension_axes", []),
        "candidate_extension_mechanisms": extension_policy.get("candidate_extension_mechanisms", []),
        "mechanism_extension_policy": extension_policy,
        "forbidden_added_mechanisms": extension_policy["hard_forbidden_mechanisms"],
        "scope_boundary": (
            "Preserve the detected wireless technologies, scenario, task, assumptions, and KPIs; "
            "only concretize missing layers needed for a well-posed research problem."
        ),
        "phase2_risks": infer_phase2_risks(ontology, taxonomy_plan),
    }
    return {
        "agent_id": "scout_agent",
        "source": "deterministic_wireless_ontology_and_taxonomy",
        "wireless_ontology": ontology,
        "taxonomy_plan": taxonomy_plan,
        "scope_contract": scope_contract,
        "generated_at": utcnow_iso(),
}


def build_mechanism_extension_policy(topic: str, preserved_tags: list[str] | set[str]) -> dict[str, Any]:
    hard_forbidden = infer_forbidden_added_mechanisms(preserved_tags)
    if not should_consider_mechanism_extensions(topic, preserved_tags):
        return {
            "enabled": False,
            "reason": "The topic is already narrow enough; do not add absent wireless mechanisms.",
            "candidate_extension_axes": [],
            "candidate_extension_mechanisms": [],
            "hard_forbidden_mechanisms": hard_forbidden,
            "selection_rules": [
                "Preserve the user topic exactly.",
                "Do not add a wireless technology solely to create novelty.",
            ],
        }

    axes = build_high_level_extension_axes(topic, preserved_tags)
    return {
        "enabled": True,
        "reason": (
            "The topic is broad enough that Phase 1 may ask the LLM to propose concrete "
            "gap-driven candidates under high-level research axes."
        ),
        "candidate_extension_axes": axes,
        "candidate_extension_mechanisms": [],
        "hard_forbidden_mechanisms": hard_forbidden if "near_field" in set(preserved_tags) else [],
        "selection_rules": [
            "Always include one plain in-scope candidate without added wireless technology.",
            "At most two candidates may add one LLM-selected concrete mechanism.",
            "The controller supplies high-level axes only; the LLM must select the concrete mechanism and justify it from topic/evidence/gap reasoning.",
            "A combination is allowed only if the selected mechanism changes the resource coupling mechanism, not just the title.",
            "Neither plain nor extension candidates win by default; choose the strongest research gap and optimization contribution.",
            "An extension must outperform the plain formulation on gap specificity, mechanism coupling, evidence plausibility, Phase 2 implementability, and experiment observability.",
            "Do not choose a model-fidelity or device-response tweak merely because it is familiar or easy to simulate.",
            "For a broad single-technology topic, include at least one candidate that adds a concrete controllable layer from the open axes, such as resource granularity, hardware response, uncertainty, coordination, or deployment geometry; reject standard objective-only formulations unless their gap is genuinely specific.",
            "Prefer tractable framework, algorithm, operating-regime insight, and paper-level numerical evidence over hard exactness claims.",
            "Reject combinations that make Phase 2 require a difficult theorem, unstable experiment, or uncontrolled model stack.",
        ],
        "axis_selection_policy": "Concrete mechanisms are selected by the LLM, not by controller enumeration.",
    }


def build_high_level_extension_axes(topic: str, preserved_tags: list[str] | set[str]) -> list[dict[str, str]]:
    """Return general research axes; never preselect concrete technologies."""

    _ = topic, preserved_tags
    return [
        {
            "id": "propagation_or_spatial_geometry",
            "label": "propagation / spatial geometry",
            "research_question": "Would changing the controllable propagation or spatial coupling create a new joint resource allocation tradeoff?",
            "selection_test": "The concrete mechanism must change channel geometry, spatial alignment, coverage, or sensing/powering coupling.",
        },
        {
            "id": "mobility_or_deployment",
            "label": "mobility / deployment",
            "research_question": "Would adding placement, trajectory, association, or deployment geometry create an optimization variable with physical value?",
            "selection_test": "The concrete mechanism must create a tractable placement, trajectory, topology, or association decision.",
        },
        {
            "id": "uncertainty_or_reliability",
            "label": "uncertainty / reliability",
            "research_question": "Would imperfect information, robustness, outage, or reliability requirements expose a meaningful operating regime?",
            "selection_test": "The concrete mechanism must introduce a well-defined uncertainty/reliability model and measurable KPI impact.",
        },
        {
            "id": "distributed_coordination",
            "label": "distributed coordination / topology",
            "research_question": "Would distributed nodes, cooperation, fronthaul, association, or coordination change the feasible set or resource coupling?",
            "selection_test": "The concrete mechanism must add a real coordination variable or constraint, not only more nodes.",
        },
        {
            "id": "access_or_resource_granularity",
            "label": "access / resource granularity",
            "research_question": "Would scheduling, multiple access, waveform, subcarrier, time, or stream granularity change the optimization structure?",
            "selection_test": "The concrete mechanism must create a controllable allocation dimension and paper-level KPI insight.",
        },
        {
            "id": "energy_or_hardware_response",
            "label": "energy / hardware response",
            "research_question": "Would a device, energy, RF-chain, or hardware response be the actual bottleneck rather than only a modeling detail?",
            "selection_test": "The concrete mechanism must alter the optimization-relevant marginal value or feasible set, not merely replace one response curve with another.",
        },
    ]


def should_consider_mechanism_extensions(topic: str, preserved_tags: list[str] | set[str]) -> bool:
    present = set(preserved_tags)
    topic_lower = topic.lower()
    already_narrow = {
        "ris",
        "uav_aided",
        "nonlinear_eh_model",
        "imperfect_csi",
        "cell_free",
        "ofdma",
        "noma",
        "security_pls",
        "thz",
    }
    if present & already_narrow:
        return False
    broad_research_tags = {
        "isacp",
        "isac",
        "swipt",
        "wpt",
        "resource_allocation",
        "power_allocation",
        "beamforming_precoding",
        "covariance_design",
        "scheduling",
        "interference_management",
        "sum_rate",
        "energy_efficiency",
        "harvested_power",
        "transmit_power",
    }
    has_broad_wireless_signal = bool(present & broad_research_tags) or looks_like_wireless_topic(topic)
    if not has_broad_wireless_signal:
        return False
    narrow_words = (
        "ris",
        "reconfigurable",
        "uav",
        "drone",
        "nonlinear",
        "non-linear",
        "imperfect csi",
        "robust csi",
        "cell-free",
        "cell free",
        "ofdm",
        "ofdma",
        "noma",
        "secrecy",
        "secure",
        "thz",
        "terahertz",
    )
    return not any(word in topic_lower for word in narrow_words)


def infer_forbidden_added_mechanisms(preserved_tags: list[str] | set[str]) -> list[str]:
    present = set(preserved_tags)
    mechanism_tags = {
        "ris": "RIS",
        "uav_aided": "UAV",
        "swipt": "SWIPT",
        "isac": "ISAC",
        "security_pls": "physical-layer security",
        "noma": "NOMA",
        "ofdma": "OFDM/OFDMA",
        "thz": "THz",
        "cell_free": "cell-free",
        "near_field": "near-field",
        "imperfect_csi": "robust/imperfect CSI",
        "nonlinear_eh_model": "nonlinear energy harvesting",
    }
    if "isacp" in present:
        present.update({"isac", "wpt", "swipt"})
    return [label for tag, label in mechanism_tags.items() if tag not in present]


def infer_phase2_risks(ontology: dict[str, Any], taxonomy_plan: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    taxonomy_profile = dict(taxonomy_plan.get("input_profile") or {}) if isinstance(taxonomy_plan, dict) else {}
    taxonomy_layers = taxonomy_profile.get("layers", {}) if isinstance(taxonomy_profile, dict) else {}
    has_taxonomy_signal = any(bool(tags) for tags in taxonomy_layers.values() if isinstance(tags, list))
    if not ontology.get("is_wireless") and not has_taxonomy_signal:
        risks.append("topic may not contain enough wireless-specific signal")
    missing_layers = taxonomy_plan.get("missing_layers", []) if isinstance(taxonomy_plan, dict) else []
    if missing_layers:
        risks.append("wireless taxonomy has underspecified layers: " + ", ".join(str(item) for item in missing_layers))
    if not ontology.get("layers", {}).get("metrics"):
        risks.append("primary system KPI must be chosen before experiments")
    if not ontology.get("layers", {}).get("solver_family"):
        risks.append("theory/algorithm route is not explicit in the topic")
    return risks or ["downstream phases must verify novelty and citations before writing claims"]


def validate_research_object_payload(payload: dict[str, Any]) -> None:
    required = (
        "topic_profile",
        "research_object",
        "wireless_system_seed",
        "mechanism_hypothesis",
        "phase2_readiness",
        "direction_constraints",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("research object payload missing keys: " + ", ".join(missing))
    research_object = dict(payload.get("research_object") or {})
    for key in ("research_question", "physical_mechanism", "expected_research_gain"):
        if not str(research_object.get(key) or "").strip():
            raise ValueError(f"research_object.{key} is required")


def build_research_frame_payload(
    topic: str,
    scope_context: dict[str, Any],
    research_payload: dict[str, Any],
) -> dict[str, Any]:
    topic_profile = dict(research_payload.get("topic_profile") or {})
    scope_contract = dict(scope_context.get("scope_contract") or {})
    for key, value in scope_contract.items():
        topic_profile.setdefault(key, value)
    return {
        "source_topic": topic,
        "topic_profile": topic_profile,
        "wireless_scope": scope_context,
        "research_object": research_payload.get("research_object", {}),
        "wireless_system_seed": research_payload.get("wireless_system_seed", {}),
        "mechanism_hypothesis": research_payload.get("mechanism_hypothesis", {}),
        "phase2_readiness": research_payload.get("phase2_readiness", {}),
        "direction_constraints": research_payload.get("direction_constraints", {}),
        "generated_at": utcnow_iso(),
    }


def normalize_direction_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_contract_payload(dict(payload))
    candidates = coerce_list(normalized.get("candidate_directions"))
    selected = dict(normalized.get("selected_candidate") or {})
    selected_title = str(selected.get("title") or "").strip()
    if not candidates and selected_title:
        candidates = [
            {
                "id": "selected",
                "title": selected_title,
                "problem_statement": selected.get("problem_statement", ""),
                "wireless_scenario": selected.get("wireless_scenario", ""),
                "research_angle": selected.get("claimed_contribution", ""),
                "mechanism_for_gain": selected.get("expected_research_gain", ""),
                "mechanism_interaction": selected.get("expected_research_gain", ""),
                "resource_coupling_change": selected.get("resource_coupling_change")
                or selected.get("source_of_difficulty")
                or selected.get("source_of_nonconvexity", ""),
                "expected_kpi_gain": selected.get("expected_research_gain", ""),
                "operating_regime": dict(normalized.get("validation_contract") or {}).get("expected_trends", ""),
                "tractability_risk": dict(normalized.get("novelty_contract") or {}).get("main_risk", ""),
                "combination_novelty": dict(normalized.get("novelty_contract") or {}).get("novelty_hypothesis", ""),
                "why_not_keyword_stacking": dict(normalized.get("novelty_contract") or {}).get("claim_boundary", ""),
                "new_coupling_or_tradeoff": selected.get("new_coupling_or_tradeoff")
                or selected.get("source_of_difficulty")
                or selected.get("source_of_nonconvexity", ""),
                "performance_bottleneck_addressed": dict(normalized.get("novelty_contract") or {}).get("prior_art_boundary", ""),
                "testable_gain_regime": dict(normalized.get("validation_contract") or {}).get("expected_trends", ""),
                "optimization_gap": dict(normalized.get("novelty_contract") or {}).get("optimization_novelty", ""),
                "optimization_novelty": dict(normalized.get("novelty_contract") or {}).get("optimization_novelty", ""),
                "objective_constraint_structure": dict(normalized.get("novelty_contract") or {}).get("objective_constraint_delta", ""),
                "algorithmic_route": dict(normalized.get("novelty_contract") or {}).get("algorithmic_delta", ""),
                "evidence_alignment": dict(normalized.get("novelty_contract") or {}).get("prior_art_boundary", ""),
                "phase2_risks": dict(normalized.get("novelty_contract") or {}).get("main_risk", ""),
                "kill_criteria": normalized.get("kill_criteria", []),
            }
        ]
    normalized["candidate_directions"] = candidates

    decision = dict(normalized.get("selection_decision") or {})
    candidate_ids = [str(dict(item or {}).get("id") or "").strip() for item in candidates if isinstance(item, dict)]
    if not decision:
        selected_id = candidate_ids[0] if candidate_ids else "selected"
        decision = {
            "selected_id": selected_id,
            "selected_title": selected_title,
            "rationale": "Selected by ScoutAgent direction review.",
            "rejected_ids": [item for item in candidate_ids if item != selected_id],
            "readiness_score_1_to_10": 8.0,
        }
    else:
        decision.setdefault("selected_title", selected_title)
        if not decision.get("selected_id") and candidate_ids:
            decision["selected_id"] = candidate_ids[0]
        selected_id = str(decision.get("selected_id") or "")
        decision.setdefault("rejected_ids", [item for item in candidate_ids if item != selected_id])
    normalized["selection_decision"] = decision
    return normalized


def validate_direction_contract_payload(payload: dict[str, Any]) -> None:
    validate_handoff_payload(payload)
    candidates = coerce_list(payload.get("candidate_directions"))
    if not candidates:
        raise ValueError("direction contract payload has no candidate_directions")
    for index, candidate in enumerate(candidates, start=1):
        cand = dict(candidate or {})
        for key in ("id", "title", "problem_statement", "kill_criteria"):
            if not cand.get(key):
                raise ValueError(f"candidate_directions[{index}].{key} is required")
        validate_candidate_mechanism_logic(cand, index=index)
    decision = dict(payload.get("selection_decision") or {})
    if not str(decision.get("selected_title") or "").strip():
        raise ValueError("selection_decision.selected_title is required")


def validate_candidate_mechanism_logic(candidate: dict[str, Any], *, index: int) -> None:
    title = str(candidate.get("title") or "")
    text = " ".join(
        str(candidate.get(key) or "")
        for key in (
            "title",
            "problem_statement",
            "research_angle",
            "mechanism_for_gain",
            "mechanism_interaction",
            "resource_coupling_change",
            "expected_kpi_gain",
            "operating_regime",
        )
    ).lower()
    extension_markers = (
        "nonlinear",
        "non-linear",
        "ris",
        "reconfigurable",
        "uav",
        "imperfect csi",
        "robust",
        "cell-free",
        "cell free",
    )
    is_extension = any(marker in text for marker in extension_markers)
    if not is_extension:
        return
    required = (
        "mechanism_interaction",
        "resource_coupling_change",
        "expected_kpi_gain",
        "operating_regime",
        "tractability_risk",
        "combination_novelty",
        "why_not_keyword_stacking",
        "new_coupling_or_tradeoff",
        "performance_bottleneck_addressed",
        "testable_gain_regime",
        "optimization_gap",
        "optimization_novelty",
        "objective_constraint_structure",
        "algorithmic_route",
    )
    missing = [key for key in required if not str(candidate.get(key) or "").strip()]
    if missing:
        raise ValueError(f"candidate_directions[{index}] missing mechanism logic keys: {', '.join(missing)}")
    combined = " ".join(str(candidate.get(key) or "") for key in required).lower()
    weak_markers = ("tbd", "n/a", "not specified", "generic", "improves performance", "better performance")
    if any(marker in combined for marker in weak_markers):
        warnings = coerce_list(candidate.get("mechanism_logic_warnings"))
        warnings.append(
            "Candidate mechanism logic contains broad performance language; downstream agents should make the gain mechanism concrete."
        )
        candidate["mechanism_logic_warnings"] = warnings


def validate_handoff_payload(payload: dict[str, Any]) -> None:
    required = (
        "selected_candidate",
        "problem_contract_seed",
        "novelty_contract",
        "proof_contract",
        "validation_contract",
        "kill_criteria",
    )
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError("handoff payload missing keys: " + ", ".join(missing))
    selected = dict(payload.get("selected_candidate") or {})
    selected_required = (
        "title",
        "problem_statement",
        "wireless_scenario",
        "objective",
        "claimed_contribution",
        "novelty_delta",
    )
    missing_selected = [key for key in selected_required if not str(selected.get(key) or "").strip()]
    if missing_selected:
        raise ValueError("selected_candidate missing keys: " + ", ".join(missing_selected))
    for key in ("problem_contract_seed", "novelty_contract", "proof_contract", "validation_contract"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"{key} must be an object")
    if not coerce_list(payload.get("kill_criteria")):
        raise ValueError("kill_criteria must be a non-empty array")


def normalize_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    selected = dict(normalized.get("selected_candidate") or {})
    problem = dict(normalized.get("problem_contract_seed") or {})
    validation = dict(normalized.get("validation_contract") or {})

    controls = coerce_list(problem.get("controls") or problem.get("variables") or selected.get("variables") or selected.get("controls"))
    if controls:
        problem.setdefault("controls", controls)
        problem.setdefault("variables", controls)
        selected.setdefault("variables", controls)
    if problem.get("constraints") and not selected.get("core_constraints"):
        selected["core_constraints"] = problem.get("constraints")
    if problem.get("objective") and not selected.get("objective"):
        selected["objective"] = problem.get("objective")
    if selected.get("source_of_nonconvexity") and not selected.get("source_of_difficulty"):
        selected["source_of_difficulty"] = selected.get("source_of_nonconvexity")
    if selected.get("convexification_path") and not selected.get("tractability_path"):
        selected["tractability_path"] = selected.get("convexification_path")
    if not selected.get("optimization_structure"):
        selected["optimization_structure"] = dict(normalized.get("novelty_contract") or {}).get(
            "objective_constraint_delta",
            selected.get("objective", ""),
        )
    if validation.get("metrics") and not problem.get("primary_kpis"):
        problem["primary_kpis"] = validation.get("metrics")
    normalized["selected_candidate"] = selected
    normalized["problem_contract_seed"] = problem
    normalized["validation_contract"] = validation
    normalized.setdefault("handoff_notes", {})
    return normalized


def finalize_handoff_payload(
    topic: str,
    contract_payload: dict[str, Any],
    evidence_pack: dict[str, Any],
    candidate_review: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    payload = dict(contract_payload)
    payload.setdefault("handoff_notes", {})
    payload["source_topic"] = topic
    payload["source_phase1_run"] = str(run_dir)
    payload["generated_at"] = utcnow_iso()
    payload["phase1_design"] = "wara_native_4_phase_controller"
    payload["evidence_summary"] = {
        "citation_policy": evidence_pack.get("citation_policy", "Do not fabricate citations; verify references downstream."),
        "reference_search_queries": evidence_pack.get("reference_search_queries", []),
        "grounded_reference_count": len(coerce_list(evidence_pack.get("references"))),
        "literature_card_count": len(coerce_list(evidence_pack.get("literature_cards"))),
        "abstract_or_pdf_card_count": int(
            dict(evidence_pack.get("literature_evidence_summary") or {}).get("abstract_or_pdf_backed_cards") or 0
        ),
        "paper_reading_record_count": len(coerce_list(evidence_pack.get("paper_reading_records"))),
        "gap_signal_count": len(coerce_list(evidence_pack.get("gap_signals"))),
        "grounding_policy": (
            "Gap signals are grounded by abstract-backed or PDF-text-backed literature cards; "
            "metadata-only records are retained for references but not used as gap evidence."
        ),
        "search_mode": evidence_pack.get("search_mode", "local"),
    }
    payload["candidate_review_summary"] = {
        "selected_title": dict(candidate_review.get("selection_decision") or {}).get("selected_title"),
        "readiness_score_1_to_10": dict(candidate_review.get("selection_decision") or {}).get("readiness_score_1_to_10"),
        "gate_scores": candidate_review.get("gate_scores", {}),
    }
    return payload


def mirror_handoff_artifacts(
    *,
    source_run_dir: Path,
    handoff_dir: Path,
    handoff_payload: dict[str, Any],
    evidence_pack: dict[str, Any],
    candidates: list[Any],
    candidate_review: dict[str, Any],
    hypotheses_md: str,
    topic_score: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    write_text(handoff_dir / "phase1_handoff.json", dump_json(handoff_payload))
    write_text(handoff_dir / "evidence_pack.json", dump_json(evidence_pack))
    write_text(handoff_dir / "candidates.json", dump_json({"candidates": candidates}))
    write_text(handoff_dir / "candidate_review.json", dump_json(candidate_review))
    write_text(handoff_dir / "hypotheses.md", hypotheses_md)
    write_text(handoff_dir / "topic_score.json", dump_json(topic_score))
    write_text(handoff_dir / "phase1_tail_summary.json", dump_json(summary))
    write_text(handoff_dir / "topic_focused_literature.json", dump_json(normalize_topic_literature(evidence_pack)))
    write_text(handoff_dir / "topic_focused_literature.md", render_evidence_pack_markdown(evidence_pack))
    references_bib = str(evidence_pack.get("references_bib") or "")
    write_text(handoff_dir / "topic_focused_references.bib", references_bib)
    write_text(handoff_dir / "reference_map.bib", references_bib)
    write_text(source_run_dir / "phase1_handoff.json", dump_json(handoff_payload))
    write_text(source_run_dir / "phase1_tail_summary.json", dump_json(summary))


def normalize_candidate_review(
    scout_payload: dict[str, Any],
    scope_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = dict(scout_payload.get("selection_decision") or {})
    candidates = coerce_list(scout_payload.get("candidate_directions"))
    taxonomy_plan = dict((scope_context or {}).get("taxonomy_plan") or {})
    candidate_assessments = []
    for candidate in candidates:
        cand = dict(candidate or {})
        candidate_text = candidate_positive_assessment_text(cand)
        assessment = assess_wireless_topic_taxonomy_candidate(candidate_text, taxonomy_plan=taxonomy_plan)
        candidate_assessments.append(
            {
                "id": cand.get("id"),
                "title": cand.get("title"),
                "taxonomy_alignment": assessment,
                "contract_field_coverage": contract_field_coverage(cand),
            }
        )
    selected_id = str(decision.get("selected_id") or "").strip()
    selected_assessment = next(
        (item for item in candidate_assessments if str(item.get("id") or "").strip() == selected_id),
        candidate_assessments[0] if candidate_assessments else {},
    )
    coverage = float(selected_assessment.get("contract_field_coverage") or 0.0)
    taxonomy_alignment = dict(selected_assessment.get("taxonomy_alignment") or {})
    wireless_fit = float(taxonomy_alignment.get("coverage") or taxonomy_alignment.get("layer_coverage") or 0.0)
    return {
        "overall_recommendation": "proceed",
        "selection_decision": decision,
        "selection_rubric": scout_payload.get("selection_rubric", {}),
        "review_mode": "deterministic_wireless_contract_readiness",
        "candidate_assessments": candidate_assessments,
        "mechanism_logic_review": review_mechanism_logic(candidates),
        "gate_scores": {
            "contract_field_coverage": round(10.0 * coverage, 2),
            "wireless_taxonomy_coverage": round(10.0 * wireless_fit, 2),
            "selection_readiness": decision.get("readiness_score_1_to_10"),
        },
        "notes": [
            "Phase 1 selects a research direction for Phase 2 contract generation.",
            "This review checks wireless taxonomy and contract readiness; it does not choose empirical comparison methods.",
            "Novelty and citation claims remain provisional until downstream evidence checks.",
        ],
    }


def review_mechanism_logic(candidates: list[Any]) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for candidate in candidates:
        cand = dict(candidate or {})
        fields = {
            key: str(cand.get(key) or "").strip()
            for key in (
                "selected_extension_axis",
                "concrete_mechanism",
                "mechanism_interaction",
                "resource_coupling_change",
                "expected_kpi_gain",
                "operating_regime",
                "tractability_risk",
                "combination_novelty",
                "why_not_keyword_stacking",
                "new_coupling_or_tradeoff",
                "performance_bottleneck_addressed",
                "testable_gain_regime",
                "optimization_gap",
                "optimization_novelty",
                "objective_constraint_structure",
                "algorithmic_route",
            )
        }
        present = [key for key, value in fields.items() if value]
        reviews.append(
            {
                "id": cand.get("id"),
                "title": cand.get("title"),
                "required_fields_present": present,
                "passes_no_pileup_check": len(present) == len(fields),
                "missing_fields": [key for key in fields if key not in present],
            }
        )
    return reviews


def candidate_positive_assessment_text(candidate: dict[str, Any]) -> str:
    scenario = sanitize_positive_taxonomy_value(candidate.get("wireless_scenario"))
    if isinstance(scenario, dict):
        scenario = {
            key: value
            for key, value in scenario.items()
            if key not in {"excluded_scope", "forbidden_scope", "non_goals"}
        }
    preserved_scope = sanitize_positive_taxonomy_value(candidate.get("preserved_topic_scope"))
    if isinstance(preserved_scope, dict):
        preserved_scope = {
            key: value
            for key, value in preserved_scope.items()
            if not is_negative_scope_key(key)
        }
    positive_payload = {
        "title": candidate.get("title"),
        "problem_statement": candidate.get("problem_statement"),
        "wireless_scenario": scenario,
        "preserved_topic_scope": preserved_scope,
        "research_angle": sanitize_positive_taxonomy_value(candidate.get("research_angle")),
        "selected_extension_axis": sanitize_positive_taxonomy_value(candidate.get("selected_extension_axis")),
        "concrete_mechanism": sanitize_positive_taxonomy_value(candidate.get("concrete_mechanism")),
        "mechanism_for_gain": sanitize_positive_taxonomy_value(candidate.get("mechanism_for_gain")),
        "mechanism_interaction": sanitize_positive_taxonomy_value(candidate.get("mechanism_interaction")),
        "resource_coupling_change": sanitize_positive_taxonomy_value(candidate.get("resource_coupling_change")),
        "expected_kpi_gain": sanitize_positive_taxonomy_value(candidate.get("expected_kpi_gain")),
        "operating_regime": sanitize_positive_taxonomy_value(candidate.get("operating_regime")),
        "tractability_risk": sanitize_positive_taxonomy_value(candidate.get("tractability_risk")),
        "combination_novelty": sanitize_positive_taxonomy_value(candidate.get("combination_novelty")),
        "why_not_keyword_stacking": sanitize_positive_taxonomy_value(candidate.get("why_not_keyword_stacking")),
        "new_coupling_or_tradeoff": sanitize_positive_taxonomy_value(candidate.get("new_coupling_or_tradeoff")),
        "performance_bottleneck_addressed": sanitize_positive_taxonomy_value(candidate.get("performance_bottleneck_addressed")),
        "testable_gain_regime": sanitize_positive_taxonomy_value(candidate.get("testable_gain_regime")),
        "optimization_gap": sanitize_positive_taxonomy_value(candidate.get("optimization_gap")),
        "optimization_novelty": sanitize_positive_taxonomy_value(candidate.get("optimization_novelty")),
        "objective_constraint_structure": sanitize_positive_taxonomy_value(candidate.get("objective_constraint_structure")),
        "algorithmic_route": sanitize_positive_taxonomy_value(candidate.get("algorithmic_route")),
        "evidence_alignment": sanitize_positive_taxonomy_value(candidate.get("evidence_alignment")),
        "controls": sanitize_positive_taxonomy_value(candidate.get("controls")),
        "parameters": sanitize_positive_taxonomy_value(candidate.get("parameters")),
        "derived_quantities": sanitize_positive_taxonomy_value(candidate.get("derived_quantities")),
        "objective": sanitize_positive_taxonomy_value(candidate.get("objective")),
        "constraints": sanitize_positive_taxonomy_value(candidate.get("constraints")),
        "expected_research_gain": sanitize_positive_taxonomy_value(candidate.get("expected_research_gain")),
        "theoretical_route": sanitize_positive_taxonomy_value(candidate.get("theoretical_route")),
        "algorithm_route": sanitize_positive_taxonomy_value(candidate.get("algorithm_route")),
        "validation_metrics": sanitize_positive_taxonomy_value(candidate.get("validation_metrics")),
        "novelty_hypothesis": sanitize_positive_taxonomy_value(candidate.get("novelty_hypothesis")),
    }
    return json.dumps(positive_payload, ensure_ascii=False)


def sanitize_positive_taxonomy_value(value: Any) -> Any:
    if isinstance(value, str):
        text = re.sub(
            r"\b(?:no|without|excluding|exclude|excluded|forbidden|must not add|do not add|not)\b[^.;]*",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", text).strip()
    if isinstance(value, list):
        return [sanitize_positive_taxonomy_value(item) for item in value if not looks_negative_text(item)]
    if isinstance(value, dict):
        return {
            key: sanitize_positive_taxonomy_value(item)
            for key, item in value.items()
            if not is_negative_scope_key(key) and not looks_negative_text(item)
        }
    return value


def looks_negative_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.search(r"\b(no|without|excluding|excluded|forbidden|must not|do not add|not)\b", text))


def is_negative_scope_key(key: Any) -> bool:
    text = str(key or "").lower()
    return any(marker in text for marker in ("not_added", "excluded", "forbidden", "must_not_add", "non_goal"))


def contract_field_coverage(candidate: dict[str, Any]) -> float:
    if candidate.get("objective") or candidate.get("controls") or candidate.get("validation_metrics"):
        required = (
            "title",
            "problem_statement",
            "wireless_scenario",
            "controls",
            "parameters",
            "derived_quantities",
            "objective",
            "constraints",
            "theoretical_route",
            "algorithm_route",
            "validation_metrics",
            "kill_criteria",
        )
    else:
        required = (
            "id",
            "title",
            "problem_statement",
            "wireless_scenario",
            "research_angle",
            "selected_extension_axis",
            "concrete_mechanism",
            "mechanism_for_gain",
            "mechanism_interaction",
            "resource_coupling_change",
            "expected_kpi_gain",
            "operating_regime",
            "tractability_risk",
            "combination_novelty",
            "why_not_keyword_stacking",
            "new_coupling_or_tradeoff",
            "performance_bottleneck_addressed",
            "testable_gain_regime",
            "optimization_gap",
            "optimization_novelty",
            "objective_constraint_structure",
            "algorithmic_route",
            "evidence_alignment",
            "phase2_risks",
            "kill_criteria",
        )
    present = sum(1 for key in required if candidate.get(key))
    return round(present / len(required), 3)


def build_grounded_evidence_pack(
    topic: str,
    scope_context: dict[str, Any],
    research_payload: dict[str, Any],
    scout_payload: dict[str, Any] | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    selected = selected_candidate_from_scout(scout_payload or {})
    queries = build_reference_search_queries(topic, research_payload, selected)
    evidence_needed = coerce_list(dict(research_payload.get("mechanism_hypothesis") or {}).get("evidence_needed"))
    evidence_needed.extend(coerce_list(selected.get("evidence_questions")))

    seminal = filter_seminal_matches(
        load_seminal_papers(" ".join([topic, str(selected.get("title") or "")])),
        " ".join([topic, str(selected.get("title") or "")]),
    )
    search_enabled = phase1_literature_search_enabled()
    literature_sources = resolve_phase1_literature_sources()
    search_errors: list[str] = []
    papers = []
    actual_source_stats: dict[str, int] = {}
    source_errors: dict[str, str] = {}
    cache_hits = 0
    if search_enabled and queries:
        try:
            from wara_core.literature.search import papers_to_bibtex, search_papers_multi_query

            query_limit = max(3, int(os.environ.get("WARA_PHASE1_LITERATURE_QUERY_LIMIT", "8")))
            papers = search_papers_multi_query(
                queries[:query_limit],
                limit_per_query=int(os.environ.get("WARA_PHASE1_LITERATURE_LIMIT", "8")),
                sources=literature_sources,
                year_min=int(os.environ.get("WARA_PHASE1_LITERATURE_YEAR_MIN", "2015")),
                inter_query_delay=float(os.environ.get("WARA_PHASE1_LITERATURE_DELAY", "0.25")),
            )
            raw_paper_count = len(papers)
            literature_filter_payload = merge_scope_profile_for_literature_filter(research_payload, scope_context, selected)
            papers = filter_relevant_literature_papers(
                papers,
                topic=topic,
                research_payload=literature_filter_payload,
                selected=selected,
                queries=queries,
            )
            papers = enrich_literature_papers_with_readable_content(
                papers,
                topic=topic,
                queries=queries,
            )
            stats = getattr(search_papers_multi_query, "last_source_stats", {})
            if isinstance(stats, dict):
                actual_source_stats = {str(key): int(value or 0) for key, value in stats.items()}
            errors = getattr(search_papers_multi_query, "last_source_errors", {})
            if isinstance(errors, dict):
                source_errors = {str(key): str(value) for key, value in errors.items()}
            cache_hits = int(getattr(search_papers_multi_query, "last_cache_hits", 0) or 0)
            filtered_out_count = max(0, raw_paper_count - len(papers))
            minimum_abstract_cards = phase1_minimum_abstract_cards(search_enabled=True)
            if count_readable_papers(papers) + len(seminal) < minimum_abstract_cards:
                abstract_queries = build_reference_supplemental_queries(topic, research_payload, selected, queries)
                abstract_sources = phase1_abstract_literature_sources(literature_sources)
                if abstract_queries and abstract_sources:
                    abstract_papers = search_papers_multi_query(
                        abstract_queries[: max(2, int(os.environ.get("WARA_PHASE1_ABSTRACT_QUERY_LIMIT", "4") or 4))],
                        limit_per_query=int(os.environ.get("WARA_PHASE1_ABSTRACT_SEARCH_LIMIT", "8") or 8),
                        sources=abstract_sources,
                        year_min=int(os.environ.get("WARA_PHASE1_LITERATURE_YEAR_MIN", "2015")),
                        inter_query_delay=float(os.environ.get("WARA_PHASE1_ABSTRACT_SEARCH_DELAY", "0.35")),
                    )
                    abstract_raw_count = len(abstract_papers)
                    abstract_papers = filter_relevant_literature_papers(
                        abstract_papers,
                        topic=topic,
                        research_payload=literature_filter_payload,
                        selected=selected,
                        queries=queries + abstract_queries,
                    )
                    abstract_papers = enrich_literature_papers_with_readable_content(
                        abstract_papers,
                        topic=topic,
                        queries=queries + abstract_queries,
                    )
                    papers = dedupe_literature_papers_prefer_readable([*papers, *abstract_papers])
                    filtered_out_count += max(0, abstract_raw_count - len(abstract_papers))
                    abstract_stats = getattr(search_papers_multi_query, "last_source_stats", {})
                    if isinstance(abstract_stats, dict):
                        for key, value in abstract_stats.items():
                            actual_source_stats[str(key)] = actual_source_stats.get(str(key), 0) + int(value or 0)
                    abstract_errors = getattr(search_papers_multi_query, "last_source_errors", {})
                    if isinstance(abstract_errors, dict):
                        for key, value in abstract_errors.items():
                            source_errors[str(key)] = str(value)
                    cache_hits += int(getattr(search_papers_multi_query, "last_cache_hits", 0) or 0)
            references_bib = papers_to_bibtex(papers) if papers else ""
            minimum_reference_target = int(os.environ.get("WARA_PHASE1_REFERENCE_MIN", "12") or 12)
            if len(papers) + len(seminal) < minimum_reference_target:
                supplemental_queries = build_reference_supplemental_queries(topic, research_payload, selected, queries)
                supplemental_sources = resolve_phase1_supplemental_literature_sources(literature_sources)
                if supplemental_queries and supplemental_sources:
                    supplemental_papers = search_papers_multi_query(
                        supplemental_queries,
                        limit_per_query=int(os.environ.get("WARA_PHASE1_SUPPLEMENTAL_LITERATURE_LIMIT", "10")),
                        sources=supplemental_sources,
                        year_min=int(os.environ.get("WARA_PHASE1_LITERATURE_YEAR_MIN", "2015")),
                        inter_query_delay=float(os.environ.get("WARA_PHASE1_SUPPLEMENTAL_LITERATURE_DELAY", "0.35")),
                    )
                    supplemental_raw_count = len(supplemental_papers)
                    supplemental_papers = filter_relevant_literature_papers(
                        supplemental_papers,
                        topic=topic,
                        research_payload=literature_filter_payload,
                        selected=selected,
                        queries=queries + supplemental_queries,
                    )
                    supplemental_papers = enrich_literature_papers_with_readable_content(
                        supplemental_papers,
                        topic=topic,
                        queries=queries + supplemental_queries,
                    )
                    papers = dedupe_literature_papers_prefer_readable([*papers, *supplemental_papers])
                    references_bib = combine_bibtex_blocks(references_bib, papers_to_bibtex(supplemental_papers) if supplemental_papers else "")
                    filtered_out_count += max(0, supplemental_raw_count - len(supplemental_papers))
                    supplemental_stats = getattr(search_papers_multi_query, "last_source_stats", {})
                    if isinstance(supplemental_stats, dict):
                        for key, value in supplemental_stats.items():
                            actual_source_stats[str(key)] = actual_source_stats.get(str(key), 0) + int(value or 0)
                    supplemental_errors = getattr(search_papers_multi_query, "last_source_errors", {})
                    if isinstance(supplemental_errors, dict):
                        for key, value in supplemental_errors.items():
                            source_errors[str(key)] = str(value)
                    cache_hits += int(getattr(search_papers_multi_query, "last_cache_hits", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            search_errors.append(str(exc))
            references_bib = ""
            filtered_out_count = 0
    else:
        references_bib = ""
        filtered_out_count = 0

    search_references = paper_refs_from_search(papers)
    seminal_references = seminal_refs_from_matches(seminal)
    references = dedupe_reference_records([*search_references, *seminal_references])
    references_bib = combine_bibtex_blocks(references_bib, seminal_matches_to_bibtex(seminal))
    literature_cards, paper_reading_records = build_literature_cards(
        papers,
        seminal,
        artifact_dir=artifact_dir,
    )
    gap_signals = build_literature_gap_signals(
        topic=topic,
        research_payload=research_payload,
        selected=selected,
        literature_cards=literature_cards,
        evidence_needed=evidence_needed,
    )
    literature_evidence_summary = summarize_literature_cards(literature_cards, gap_signals)
    return {
        "source_topic": topic,
        "agent_id": "literature_agent",
        "search_mode": "external_literature_search" if search_enabled else "local_seminal_and_query_plan",
        "wireless_scope_summary": dict(scope_context.get("wireless_ontology") or {}).get("summary", ""),
        "literature_questions": build_literature_questions(research_payload, selected),
        "evidence_needed": dedupe_texts(evidence_needed),
        "reference_search_queries": queries,
        "literature_sources": list(literature_sources),
        "arxiv_usage_policy": (
            "Use arXiv as a content-discovery source for abstracts/technical context; "
            "Phase 3.3 must prefer formally published IEEE/journal/conference metadata in the final bibliography when a verified version exists."
        ),
        "source_status": build_literature_source_status(literature_sources, actual_source_stats),
        "source_result_counts": actual_source_stats,
        "source_errors": source_errors,
        "cache_hits": cache_hits,
        "filtered_out_count": filtered_out_count,
        "seminal_matches": seminal,
        "retrieved_references": search_references,
        "references": references,
        "references_bib": references_bib,
        "literature_cards": literature_cards,
        "paper_reading_records": paper_reading_records,
        "gap_signals": gap_signals,
        "literature_evidence_summary": literature_evidence_summary,
        "literature_grounding_limitations": build_literature_grounding_limitations(literature_cards),
        "search_errors": search_errors,
        "citation_policy": "Use only verified references; do not fabricate citations or priority claims.",
        "generated_at": utcnow_iso(),
    }


def phase1_literature_search_enabled() -> bool:
    raw_value = os.environ.get("WARA_PHASE1_LITERATURE_SEARCH")
    if raw_value is None:
        return True
    return raw_value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def merge_evidence_packs(base: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    """Preserve broad evidence while adding selected-direction evidence."""
    merged = dict(selected or {})
    base = dict(base or {})
    selected = dict(selected or {})

    merged["reference_search_queries"] = dedupe_texts(
        coerce_list(base.get("reference_search_queries")) + coerce_list(selected.get("reference_search_queries"))
    )
    merged["literature_questions"] = dedupe_texts(
        coerce_list(base.get("literature_questions")) + coerce_list(selected.get("literature_questions"))
    )
    merged["evidence_needed"] = dedupe_texts(coerce_list(base.get("evidence_needed")) + coerce_list(selected.get("evidence_needed")))
    merged["literature_sources"] = dedupe_texts(coerce_list(base.get("literature_sources")) + coerce_list(selected.get("literature_sources")))
    merged["seminal_matches"] = dedupe_reference_records(coerce_list(base.get("seminal_matches")) + coerce_list(selected.get("seminal_matches")))
    merged["retrieved_references"] = dedupe_reference_records(
        coerce_list(base.get("retrieved_references")) + coerce_list(selected.get("retrieved_references"))
    )
    merged["references"] = dedupe_reference_records(coerce_list(base.get("references")) + coerce_list(selected.get("references")))
    merged["references_bib"] = combine_bibtex_blocks(str(base.get("references_bib") or ""), str(selected.get("references_bib") or ""))
    merged["literature_cards"] = dedupe_structured_records(
        coerce_list(base.get("literature_cards")) + coerce_list(selected.get("literature_cards")),
        key_fields=("card_id", "cite_key", "title"),
    )
    merged["paper_reading_records"] = dedupe_structured_records(
        coerce_list(base.get("paper_reading_records")) + coerce_list(selected.get("paper_reading_records")),
        key_fields=("pdf_url", "title", "reason"),
    )
    merged["gap_signals"] = dedupe_structured_records(
        coerce_list(base.get("gap_signals")) + coerce_list(selected.get("gap_signals")),
        key_fields=("signal_id", "statement"),
    )
    merged["literature_grounding_limitations"] = dedupe_texts(
        coerce_list(base.get("literature_grounding_limitations"))
        + coerce_list(selected.get("literature_grounding_limitations"))
    )
    merged["literature_evidence_summary"] = summarize_literature_cards(
        [dict(item) for item in coerce_list(merged.get("literature_cards")) if isinstance(item, dict)],
        [dict(item) for item in coerce_list(merged.get("gap_signals")) if isinstance(item, dict)],
    )

    source_status = dict(base.get("source_status") or {})
    source_status.update(dict(selected.get("source_status") or {}))
    merged["source_status"] = source_status

    source_result_counts: dict[str, int] = {}
    for pack in (base, selected):
        for key, value in dict(pack.get("source_result_counts") or {}).items():
            source_result_counts[str(key)] = source_result_counts.get(str(key), 0) + int(value or 0)
    merged["source_result_counts"] = source_result_counts

    source_errors = dict(base.get("source_errors") or {})
    source_errors.update(dict(selected.get("source_errors") or {}))
    merged["source_errors"] = source_errors
    merged["cache_hits"] = int(base.get("cache_hits") or 0) + int(selected.get("cache_hits") or 0)
    merged["filtered_out_count"] = int(base.get("filtered_out_count") or 0) + int(selected.get("filtered_out_count") or 0)
    merged["search_errors"] = dedupe_texts(coerce_list(base.get("search_errors")) + coerce_list(selected.get("search_errors")))
    merged["citation_policy"] = selected.get("citation_policy") or base.get("citation_policy") or "Use only verified references; do not fabricate citations or priority claims."
    merged["search_mode"] = selected.get("search_mode") or base.get("search_mode") or ""
    merged["evidence_merge_policy"] = "base_topic_evidence_plus_selected_direction_evidence"
    return merged


def resolve_phase1_literature_sources() -> tuple[str, ...]:
    override = os.environ.get("WARA_PHASE1_LITERATURE_SOURCES", "").strip()
    if override:
        return tuple(normalize_literature_source(item) for item in override.split(",") if item.strip())
    sources = ["semantic_scholar", "openalex", "crossref"]
    if os.environ.get("WARA_PHASE1_ENABLE_IEEE_XPLORE", "").strip().lower() in {"1", "true", "yes", "on"}:
        sources.insert(0, "ieee_xplore")
    if os.environ.get("WARA_PHASE1_DISABLE_OPENALEX", "").strip().lower() in {"1", "true", "yes", "on"}:
        sources = [source for source in sources if source != "openalex"]
    arxiv_toggle = os.environ.get("WARA_PHASE1_ENABLE_ARXIV")
    arxiv_enabled = (
        str(arxiv_toggle).strip().lower() in {"1", "true", "yes", "on"}
        and os.environ.get("WARA_PHASE1_DISABLE_ARXIV", "").strip().lower() not in {"1", "true", "yes", "on"}
        and os.environ.get("WCL_DISABLE_ARXIV", "").strip().lower() not in {"1", "true", "yes", "on"}
    )
    if arxiv_enabled:
        sources.append("arxiv")
    if os.environ.get("WARA_PHASE1_ENABLE_SEMANTIC_SCHOLAR", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        sources = [source for source in sources if source != "semantic_scholar"]
    if os.environ.get("WARA_PHASE1_GOOGLE_SCHOLAR", "").strip().lower() in {"1", "true", "yes", "on"}:
        sources.append("google_scholar")
    return tuple(sources)


def resolve_phase1_supplemental_literature_sources(base_sources: tuple[str, ...]) -> tuple[str, ...]:
    """Use resilient metadata sources when the first literature pass underfills."""
    resolved = [normalize_literature_source(item) for item in base_sources]
    # Avoid repeatedly hammering arXiv after a narrow query underfills or gets rate-limited.
    resolved = [item for item in resolved if item != "arxiv"]
    if "crossref" not in resolved:
        resolved.insert(0, "crossref")
    if os.environ.get("WARA_PHASE1_SUPPLEMENTAL_OPENALEX", "").strip().lower() in {"1", "true", "yes", "on"}:
        if "openalex" not in resolved:
            resolved.append("openalex")
    if os.environ.get("WARA_PHASE1_SUPPLEMENTAL_SEMANTIC_SCHOLAR", "1").strip().lower() not in {"0", "false", "no", "off"}:
        if "semantic_scholar" not in resolved:
            resolved.append("semantic_scholar")
    if (
        os.environ.get("WARA_PHASE1_ENABLE_IEEE_XPLORE", "").strip().lower() in {"1", "true", "yes", "on"}
        or os.environ.get("IEEE_XPLORE_API_KEY")
        or os.environ.get("IEEE_API_KEY")
    ):
        if "ieee_xplore" not in resolved:
            resolved.insert(0, "ieee_xplore")
    return tuple(dedupe_texts(resolved))


def normalize_literature_source(source: Any) -> str:
    value = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ieee": "ieee_xplore",
        "xplore": "ieee_xplore",
        "ieeexplore": "ieee_xplore",
        "s2": "semantic_scholar",
        "scholar": "google_scholar",
        "google": "google_scholar",
        "googlescholar": "google_scholar",
    }
    return aliases.get(value, value)


def build_literature_source_status(
    sources: tuple[str, ...],
    source_counts: dict[str, int] | None = None,
) -> dict[str, str]:
    source_counts = source_counts or {}
    status: dict[str, str] = {}
    for source in sources:
        if source == "ieee_xplore":
            if not (os.environ.get("IEEE_XPLORE_API_KEY") or os.environ.get("IEEE_API_KEY")):
                status[source] = "missing_api_key"
            elif source in source_counts:
                status[source] = "ok" if source_counts.get(source, 0) > 0 else "enabled_no_results_or_failed"
            else:
                status[source] = "enabled"
        elif source == "google_scholar":
            if source in source_counts:
                status[source] = "ok_unstable_scraping" if source_counts.get(source, 0) > 0 else "enabled_no_results_or_failed_unstable_scraping"
            else:
                status[source] = "enabled_unstable_scraping"
        else:
            if source in source_counts:
                status[source] = "ok" if source_counts.get(source, 0) > 0 else "enabled_no_results_or_failed"
            else:
                status[source] = "enabled"
    if "google_scholar" not in sources:
        status["google_scholar"] = "available_opt_in"
    return status


def selected_candidate_from_scout(scout_payload: dict[str, Any]) -> dict[str, Any]:
    decision = dict(scout_payload.get("selection_decision") or {})
    selected_id = str(decision.get("selected_id") or "").strip()
    candidates = [dict(item or {}) for item in coerce_list(scout_payload.get("candidate_directions"))]
    for candidate in candidates:
        if str(candidate.get("id") or "").strip() == selected_id:
            return candidate
    return candidates[0] if candidates else {}


def build_reference_search_queries(
    topic: str,
    research_payload: dict[str, Any],
    selected: dict[str, Any],
) -> list[str]:
    candidate_text = " ".join(
        [
            topic,
            str(selected.get("title") or ""),
            str(selected.get("problem_statement") or ""),
            str(selected.get("objective") or ""),
            str(selected.get("claimed_contribution") or ""),
            str(selected.get("expected_research_gain") or ""),
            flatten_for_query(selected.get("combination_novelty")),
            flatten_for_query(selected.get("new_coupling_or_tradeoff")),
        ]
    )
    acronym_terms = []
    if re.search(r"\bisacp\b|integrated sensing communication and powering", candidate_text, re.IGNORECASE):
        acronym_terms.append("ISACP")
    if re.search(r"\bisac\b|integrated sensing and communication", candidate_text, re.IGNORECASE):
        acronym_terms.append("ISAC")
    if re.search(r"\bcrb\b|cramer", candidate_text, re.IGNORECASE):
        acronym_terms.append("CRB")
    elif re.search(r"\bbeampattern\b|illumination", candidate_text, re.IGNORECASE):
        acronym_terms.append("beampattern")
    if re.search(r"\bsdr\b|semidefinite|rank", candidate_text, re.IGNORECASE):
        acronym_terms.extend(["SDR", "rank-one"])
    topic_terms = compact_query_terms(topic)
    title_terms = compact_query_terms(str(selected.get("title") or ""), max_words=6)
    kpi_terms = compact_query_terms(" ".join(str(item) for item in coerce_list(selected.get("validation_metrics"))), max_words=3)
    theory_terms = compact_query_terms(flatten_for_query(selected.get("theoretical_route")), max_words=6)
    mechanism_queries = build_mechanism_specific_queries(candidate_text, acronym_terms)
    service_axis_queries = build_service_axis_queries(candidate_text, topic_terms, title_terms, acronym_terms)
    queries = [
        " ".join(dedupe_texts(acronym_terms + [topic_terms, "wireless beamforming optimization"])),
        " ".join(dedupe_texts(acronym_terms + [title_terms, kpi_terms])),
        " ".join(dedupe_texts(acronym_terms + [topic_terms, theory_terms])),
    ] + service_axis_queries + mechanism_queries
    return dedupe_texts([query for query in queries if len(query.strip()) >= 8])[:8]


def build_reference_supplemental_queries(
    topic: str,
    research_payload: dict[str, Any],
    selected: dict[str, Any],
    primary_queries: list[str],
) -> list[str]:
    """Generate broader, mechanism-aware queries when the reference bank is underfilled.

    These are not topic-specific fallbacks.  They use mechanisms already present
    in the topic/selected direction, then broaden only the query wording so the
    LiteratureAgent can recover from API limits or overly narrow phrasing.
    """
    context = " ".join(
        [
            topic,
            str(selected.get("title") or ""),
            str(selected.get("problem_statement") or ""),
            str(selected.get("wireless_scenario") or ""),
            str(selected.get("objective") or ""),
            str(dict(research_payload.get("research_object") or {}).get("physical_mechanism") or ""),
        ]
    )
    text = context.lower()
    base = compact_query_terms(topic, max_words=8)
    title = compact_query_terms(str(selected.get("title") or ""), max_words=8)
    queries = [
        f"{base} wireless resource allocation optimization",
        f"{base} beamforming optimization",
        f"{title or base} wireless communications",
    ]
    if any(marker in text for marker in ("star-ris", "star ris", "simultaneous transmission and reflection")):
        queries.extend(
            [
                "STAR-RIS resource allocation beamforming optimization",
                "simultaneously transmitting reflecting RIS wireless communications optimization",
            ]
        )
    elif any(marker in text for marker in ("ris", "reconfigurable intelligent", "intelligent reflecting")):
        queries.extend(
            [
                "reconfigurable intelligent surface beamforming resource allocation",
                "RIS wireless communications optimization",
            ]
        )
    if any(marker in text for marker in ("full-duplex", "full duplex", "self-interference", "self interference")):
        queries.extend(
            [
                "full-duplex wireless self-interference resource allocation",
                "full-duplex communications beamforming self-interference optimization",
            ]
        )
    if any(marker in text for marker in ("near-field", "near field", "xl-mimo", "extra large")):
        queries.extend(["near-field XL-MIMO beam focusing optimization", "near-field MIMO beamforming resource allocation"])
    if any(marker in text for marker in ("movable antenna", "fluid antenna", "antenna position")):
        queries.extend(["movable antenna wireless optimization", "fluid antenna system resource allocation"])
    if any(marker in text for marker in ("uav", "drone", "unmanned aerial")):
        queries.extend(["UAV wireless communications trajectory resource allocation", "UAV beamforming optimization"])
    if any(marker in text for marker in ("robust", "imperfect csi", "channel uncertainty")):
        queries.extend(["robust wireless beamforming imperfect CSI", "distributionally robust wireless resource allocation"])
    if any(marker in text for marker in ("energy harvesting", "wireless power", "swipt", "power transfer")):
        queries.extend(["wireless power transfer beamforming optimization", "SWIPT resource allocation energy harvesting"])
    queries.extend(primary_queries[:3])
    return dedupe_texts([query for query in queries if len(query.strip()) >= 8])[:10]


def build_service_axis_queries(candidate_text: str, topic_terms: str, title_terms: str, acronym_terms: list[str]) -> list[str]:
    text = candidate_text.lower()
    prefix = " ".join(dedupe_texts(acronym_terms)) or "wireless"
    queries: list[str] = []
    if any(marker in text for marker in ("near-field", "near field", "xl-mimo", "extra large", "large-scale antenna")):
        queries.extend(
            [
                "near-field XL-MIMO beamforming optimization",
                "near-field communications survey MIMO beamforming",
            ]
        )
    if any(marker in text for marker in ("integrated sensing", "isac", "sensing")):
        queries.extend(
            [
                f"{prefix} integrated sensing communication beamforming optimization",
                f"{prefix} joint beamforming sensing communication",
            ]
        )
    if any(marker in text for marker in ("powering", "wireless power", "rf-power", "energy harvesting", "swipt")):
        queries.extend(
            [
                f"{prefix} wireless power transfer beamforming optimization",
                f"{prefix} simultaneous wireless information power transfer beamforming",
            ]
        )
    if any(marker in text for marker in ("semidefinite", "sdp", "sdr", "rank")):
        queries.append(f"{prefix} semidefinite relaxation beamforming optimization")
    if topic_terms:
        queries.append(f"{topic_terms} survey optimization")
    if title_terms:
        queries.append(f"{title_terms} related work optimization")
    return queries


def build_mechanism_specific_queries(candidate_text: str, acronym_terms: list[str]) -> list[str]:
    text = candidate_text.lower()
    queries: list[str] = []
    prefix = " ".join(dedupe_texts(acronym_terms)) or "wireless"
    if any(marker in text for marker in ("star-ris", "star ris", "simultaneous transmission and reflection")):
        queries.extend(
            [
                "STAR-RIS resource allocation beamforming optimization",
                "simultaneously transmitting reflecting RIS wireless communications optimization",
            ]
        )
    elif any(marker in text for marker in ("ris", "reconfigurable intelligent", "intelligent reflecting")):
        queries.extend(
            [
                f"{prefix} reconfigurable intelligent surface beamforming resource allocation",
                f"{prefix} RIS wireless communications optimization",
            ]
        )
    if any(marker in text for marker in ("full-duplex", "full duplex", "self-interference", "self interference")):
        queries.extend(
            [
                f"{prefix} full-duplex wireless self-interference resource allocation",
                f"{prefix} full-duplex communications beamforming optimization",
            ]
        )
    if any(marker in text for marker in ("nonlinear", "non-linear", "nonlinear-eh", "saturation")):
        queries.extend(
            [
                f"{prefix} nonlinear energy harvesting beamforming",
                f"{prefix} SWIPT nonlinear energy harvesting saturation",
                f"{prefix} energy harvesting sensitivity saturation wireless power transfer",
            ]
        )
    if any(marker in text for marker in ("uav", "drone", "unmanned aerial")):
        queries.append(f"{prefix} UAV wireless communications trajectory resource allocation")
    if any(marker in text for marker in ("robust", "imperfect csi", "channel uncertainty")):
        queries.append(f"{prefix} robust wireless beamforming imperfect CSI")
    if any(marker in text for marker in ("cell-free", "cell free")):
        queries.append(f"{prefix} cell-free wireless communications resource allocation")
    return queries


def merge_scope_profile_for_literature_filter(
    research_payload: dict[str, Any],
    scope_context: dict[str, Any],
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(research_payload)
    profile = dict(payload.get("topic_profile") or {})
    scope_contract = dict(scope_context.get("scope_contract") or {})
    for key in (
        "forbidden_added_mechanisms",
        "candidate_extension_mechanisms",
        "mechanism_extension_policy",
    ):
        if key in scope_contract:
            profile[key] = scope_contract.get(key)
    if selected:
        all_extensions = coerce_list(scope_contract.get("candidate_extension_mechanisms"))
        selected_extensions = selected_extension_mechanisms(all_extensions, selected)
        profile["candidate_extension_mechanisms"] = selected_extensions
        non_selected_extensions = [
            str(item).strip()
            for item in all_extensions
            if str(item).strip() and str(item).strip() not in set(selected_extensions)
        ]
        profile["forbidden_added_mechanisms"] = dedupe_texts(
            coerce_list(scope_contract.get("forbidden_added_mechanisms")) + non_selected_extensions
        )
    payload["topic_profile"] = profile
    return payload


def selected_extension_mechanisms(candidates: Any, selected: dict[str, Any]) -> list[str]:
    selected_text = " ".join(
        str(value or "")
        for value in (
            selected.get("title"),
            selected.get("problem_statement"),
            selected.get("wireless_scenario"),
            selected.get("claimed_contribution"),
            selected.get("expected_research_gain"),
        )
    ).lower()
    aliases = {
        "nonlinear energy harvesting": ("nonlinear", "non-linear", "nonlinear eh"),
        "RIS": ("ris", "reconfigurable intelligent surface", "intelligent reflecting surface"),
        "UAV": ("uav", "unmanned aerial", "drone"),
        "robust/imperfect CSI": ("robust", "imperfect csi", "channel uncertainty"),
        "cell-free": ("cell-free", "cell free"),
    }
    selected_candidates = []
    for candidate in coerce_list(candidates):
        label = str(candidate or "").strip()
        if not label:
            continue
        if any(alias in selected_text for alias in aliases.get(label, (label.lower(),))):
            selected_candidates.append(label)
    return selected_candidates


def filter_seminal_matches(matches: list[dict[str, Any]], context: str) -> list[dict[str, Any]]:
    context_lower = context.lower()
    filtered: list[dict[str, Any]] = []
    for paper in matches:
        keywords = paper.get("keywords", []) if isinstance(paper, dict) else []
        keep = False
        for keyword in keywords:
            kw = str(keyword or "").strip().lower()
            if not kw:
                continue
            if len(kw) <= 3:
                if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", context_lower):
                    keep = True
                    break
            elif kw in context_lower:
                keep = True
                break
        if keep:
            filtered.append(paper)
    return filtered


def filter_relevant_literature_papers(
    papers: list[Any],
    *,
    topic: str,
    research_payload: dict[str, Any],
    selected: dict[str, Any],
    queries: list[str],
) -> list[Any]:
    """Drop high-citation but off-domain hits before they pollute Phase 2."""
    if not papers:
        return []
    context = " ".join(
        [
            topic,
            str(selected.get("title") or ""),
            str(selected.get("problem_statement") or ""),
            str(dict(research_payload.get("research_object") or {}).get("physical_mechanism") or ""),
            " ".join(queries),
        ]
    )
    required_terms = extract_relevance_terms(context)
    forbidden_terms = extract_forbidden_literature_terms(topic, research_payload)
    scored: list[tuple[float, Any]] = []
    for paper in papers:
        if paper_matches_forbidden_terms(paper, forbidden_terms):
            continue
        if not paper_matches_selected_scope(paper, research_payload):
            continue
        if not paper_has_topic_cluster_overlap(paper, context):
            continue
        score = literature_relevance_score(paper, required_terms)
        if score >= 2.5:
            scored.append((score, paper))
    if not scored:
        if os.environ.get("WARA_PHASE1_ALLOW_WEAK_LITERATURE_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
            return papers[: min(len(papers), 8)]
        return []
    scored.sort(key=lambda item: (item[0], int(getattr(item[1], "citation_count", 0) or 0), int(getattr(item[1], "year", 0) or 0)), reverse=True)
    return [paper for _, paper in scored]


def wireless_reference_clusters(text: str) -> set[str]:
    lower = str(text or "").lower()
    patterns = {
        "isac": (
            "isac",
            "iscp",
            "isacp",
            "integrated sensing",
            "sensing and communication",
            "sensing communication",
        ),
        "wireless_power": (
            "wireless power",
            "power transfer",
            "powering",
            "rf power",
            "rf-power",
            "wpt",
            "energy harvesting",
            "swipt",
            "harvested energy",
        ),
        "near_field": (
            "near-field",
            "near field",
            "xl-mimo",
            "extremely large",
            "extra large",
            "holographic",
        ),
        "beamforming": (
            "beamforming",
            "precoding",
            "covariance",
            "transmit beam",
            "waveform",
        ),
        "optimization": (
            "optimization",
            "resource allocation",
            "semidefinite",
            "sdr",
            "sdp",
            "sca",
            "wmmse",
        ),
        "mimo": (
            "mimo",
            "massive mimo",
            "antenna array",
            "multi-antenna",
        ),
        "ris": (
            "ris",
            "star-ris",
            "star ris",
            "reconfigurable intelligent surface",
            "intelligent reflecting surface",
            "simultaneously transmitting",
            "transmitting and reflecting",
        ),
        "full_duplex": (
            "full-duplex",
            "full duplex",
            "self-interference",
            "self interference",
            "residual si",
        ),
        "interference_management": (
            "interference",
            "self-interference",
            "interference management",
            "resource allocation",
        ),
    }
    clusters: set[str] = set()
    for label, terms in patterns.items():
        if any(term in lower for term in terms):
            clusters.add(label)
    return clusters


def paper_has_topic_cluster_overlap(paper: Any, context: str) -> bool:
    context_clusters = wireless_reference_clusters(context)
    if not context_clusters:
        return True
    title = str(getattr(paper, "title", "") or "")
    abstract = str(getattr(paper, "abstract", "") or "")
    venue = str(getattr(paper, "venue", "") or "")
    paper_clusters = wireless_reference_clusters(" ".join([title, abstract, venue]))
    if not paper_clusters:
        return False

    service_clusters = {"isac", "wireless_power", "near_field", "mimo", "ris", "full_duplex"}
    active_service_clusters = context_clusters & service_clusters
    paper_service_clusters = paper_clusters & service_clusters
    if active_service_clusters:
        if paper_service_clusters & active_service_clusters:
            return True
        if "beamforming" in paper_clusters and paper_service_clusters:
            return True
        return False
    return bool(paper_clusters & context_clusters)


def extract_forbidden_literature_terms(topic: str, research_payload: dict[str, Any]) -> list[str]:
    topic_lower = topic.lower()
    profile = dict(research_payload.get("topic_profile") or {})
    constraints = dict(research_payload.get("direction_constraints") or {})
    raw_terms = coerce_list(profile.get("forbidden_added_mechanisms")) + coerce_list(constraints.get("must_not_add"))
    allowed_terms = {
        str(item or "").strip().lower()
        for item in coerce_list(profile.get("candidate_extension_mechanisms"))
        if str(item or "").strip()
    }
    synonyms = {
        "RIS": [
            "ris",
            "reconfigurable intelligent surface",
            "reconfigurable intelligent surfaces",
            "intelligent reflecting surface",
            "intelligent reflecting surfaces",
        ],
        "UAV": ["uav", "unmanned aerial vehicle", "drone"],
        "physical-layer security": ["physical-layer security", "physical layer security", "secure communication", "secure communications"],
        "NOMA": ["noma", "non-orthogonal multiple access"],
        "OFDM/OFDMA": ["ofdm", "ofdma", "orthogonal frequency division"],
        "THz": ["thz", "terahertz"],
        "cell-free": ["cell-free", "cell free"],
        "near-field": ["near-field", "near field"],
        "robust/imperfect CSI": ["imperfect csi", "robust csi", "channel uncertainty"],
        "nonlinear energy harvesting": ["nonlinear energy harvesting", "non-linear energy harvesting", "nonlinear eh"],
    }
    terms: list[str] = []
    for raw in raw_terms:
        label = str(raw or "").strip()
        if not label:
            continue
        if label.lower() in allowed_terms:
            continue
        for term in synonyms.get(label, [label.lower()]):
            if term.lower() not in topic_lower:
                terms.append(term.lower())
    return dedupe_texts(terms)


def paper_matches_forbidden_terms(paper: Any, forbidden_terms: list[str]) -> bool:
    if not forbidden_terms:
        return False
    title = str(getattr(paper, "title", "") or "")
    abstract = str(getattr(paper, "abstract", "") or "")
    venue = str(getattr(paper, "venue", "") or "")
    text = " ".join([title, abstract, venue]).lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) for term in forbidden_terms)


def paper_matches_selected_scope(paper: Any, research_payload: dict[str, Any]) -> bool:
    title = str(getattr(paper, "title", "") or "")
    abstract = str(getattr(paper, "abstract", "") or "")
    venue = str(getattr(paper, "venue", "") or "")
    text = " ".join([title, abstract, venue]).lower()
    off_domain_terms = (
        "ultrasound",
        "medical imaging",
        "biomedical",
        "microwave architectures",
    )
    if any(term in text for term in off_domain_terms):
        return False

    profile = dict(research_payload.get("topic_profile") or {})
    selected_extensions = {str(item).strip().lower() for item in coerce_list(profile.get("candidate_extension_mechanisms"))}
    if not selected_extensions:
        return True

    core_isac = any(
        term in text
        for term in (
            "isac",
            "isacp",
            "integrated sensing",
            "sensing and communication",
            "sensing communication",
            "monostatic sensing",
        )
    )
    core_power = any(
        term in text
        for term in (
            "energy harvesting",
            "wireless power",
            "power transfer",
            "swipt",
            "wpt",
            "powering",
            "harvested",
        )
    )
    if "nonlinear energy harvesting" in selected_extensions:
        return core_isac or core_power
    if "ris" in selected_extensions:
        return core_isac or "ris" in text or "reconfigurable intelligent" in text or "intelligent reflecting" in text
    if "uav" in selected_extensions:
        return core_isac or "uav" in text or "unmanned aerial" in text or "drone" in text
    if "robust/imperfect csi" in selected_extensions:
        return core_isac or "imperfect csi" in text or "channel uncertainty" in text or "robust" in text
    if "cell-free" in selected_extensions:
        return core_isac or "cell-free" in text or "cell free" in text
    return True


def extract_relevance_terms(context: str) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_+\-/]{2,}", context)
        if word.lower() not in {
            "the",
            "and",
            "for",
            "with",
            "under",
            "using",
            "from",
            "into",
            "phase",
            "wireless",
            "optimization",
            "communication",
            "communications",
        }
    ]
    priority = [
        "isacp",
        "isac",
        "swipt",
        "beamforming",
        "sensing",
        "powering",
        "energy",
        "harvesting",
        "mimo",
        "massive",
        "monostatic",
        "covariance",
        "sinr",
        "crb",
        "antenna",
        "ris",
        "star-ris",
        "star",
        "reconfigurable",
        "intelligent",
        "surface",
        "duplex",
        "full-duplex",
        "self-interference",
        "interference",
        "resource",
        "allocation",
        "channel",
        "transmit",
    ]
    ordered = priority + words
    return dedupe_texts(ordered)[:24]


def literature_relevance_score(paper: Any, relevance_terms: list[str]) -> float:
    title = str(getattr(paper, "title", "") or "")
    abstract = str(getattr(paper, "abstract", "") or "")
    venue = str(getattr(paper, "venue", "") or "")
    text = " ".join([title, abstract, venue]).lower()
    if not text.strip():
        return 0.0

    wireless_anchors = {
        "wireless",
        "communication",
        "communications",
        "sensing",
        "powering",
        "energy",
        "harvesting",
        "beamforming",
        "mimo",
        "antenna",
        "channel",
        "transmit",
        "sinr",
        "crb",
        "isac",
        "isacp",
        "swipt",
        "rf",
        "network",
        "6g",
        "ris",
        "star-ris",
        "reconfigurable",
        "intelligent",
        "surface",
        "full-duplex",
        "duplex",
        "self-interference",
        "interference",
        "resource allocation",
    }
    anchor_hits = sum(1 for term in wireless_anchors if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    topic_hits = sum(1 for term in relevance_terms if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text))
    phrase_bonus = 0.0
    for phrase in (
        "integrated sensing",
        "sensing communication",
        "sensing and communication",
        "communication and powering",
        "wireless power",
        "energy harvesting",
        "power transfer",
        "reconfigurable intelligent surface",
        "simultaneously transmitting",
        "full-duplex",
        "self-interference",
        "resource allocation",
    ):
        if phrase in text:
            phrase_bonus += 1.0
    title_lower = title.lower()
    title_bonus = 1.0 if any(term in title_lower for term in ("isac", "isacp", "swipt", "beamforming", "sensing", "powering", "ris", "full-duplex", "self-interference", "resource allocation")) else 0.0
    if anchor_hits == 0:
        return 0.0
    return min(float(anchor_hits), 3.0) + min(float(topic_hits) * 0.6, 4.0) + phrase_bonus + title_bonus


def compact_query_terms(text: str, *, max_words: int = 10) -> str:
    text = re.sub(r"[^A-Za-z0-9_+\-/\s]", " ", str(text or ""))
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    stop = {
        "the",
        "and",
        "with",
        "using",
        "where",
        "then",
        "under",
        "phase",
        "problem",
        "formulate",
        "design",
        "single",
        "cell",
    }
    kept = [word for word in words if word.lower() not in stop]
    return " ".join(kept[:max_words])


def flatten_for_query(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_for_query(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten_for_query(item) for item in value.values())
    return str(value or "")


def build_literature_questions(research_payload: dict[str, Any], selected: dict[str, Any]) -> list[str]:
    questions = [
        "Which prior wireless models use the same physical mechanism and assumptions?",
        "Which optimization variables and KPIs are standard for this scenario?",
        "Which theoretical routes have already been used for closely related problems?",
    ]
    mechanism = dict(research_payload.get("mechanism_hypothesis") or {})
    questions.extend(coerce_list(mechanism.get("evidence_needed")))
    questions.extend(coerce_list(selected.get("evidence_questions")))
    return dedupe_texts(questions)


def phase1_minimum_abstract_cards(*, search_enabled: bool) -> int:
    if not search_enabled:
        return int(os.environ.get("WARA_PHASE1_MIN_ABSTRACT_CARDS", "0") or 0)
    return int(os.environ.get("WARA_PHASE1_MIN_ABSTRACT_CARDS", "3") or 3)


def phase1_abstract_literature_sources(base_sources: tuple[str, ...]) -> tuple[str, ...]:
    sources = [normalize_literature_source(item) for item in base_sources]
    preferred = ["semantic_scholar", "openalex", "arxiv"]
    if (
        os.environ.get("WARA_PHASE1_ENABLE_IEEE_XPLORE", "").strip().lower() in {"1", "true", "yes", "on"}
        or os.environ.get("IEEE_XPLORE_API_KEY")
        or os.environ.get("IEEE_API_KEY")
    ):
        preferred.insert(0, "ieee_xplore")
    for source in preferred:
        if source == "arxiv" and (
            os.environ.get("WARA_PHASE1_DISABLE_ARXIV", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.environ.get("WCL_DISABLE_ARXIV", "").strip().lower() in {"1", "true", "yes", "on"}
        ):
            continue
        if source not in sources:
            sources.append(source)
    return tuple(dedupe_texts(sources))


def paper_has_readable_content(paper: Any) -> bool:
    return len(str(getattr(paper, "abstract", "") or "").strip()) >= 80


def count_readable_papers(papers: list[Any]) -> int:
    return sum(1 for paper in papers if paper_has_readable_content(paper))


def enrich_literature_papers_with_readable_content(
    papers: list[Any],
    *,
    topic: str,
    queries: list[str],
) -> list[Any]:
    """Prefer abstract-backed records and try lightweight abstract enrichment."""

    papers = dedupe_literature_papers_prefer_readable(papers)
    if not papers:
        return papers

    min_cards = phase1_minimum_abstract_cards(search_enabled=phase1_literature_search_enabled())
    if count_readable_papers(papers) >= min_cards:
        return papers

    enriched = list(papers)
    enriched.extend(fetch_semantic_scholar_details_for_missing_abstracts(papers))
    enriched.extend(fetch_arxiv_details_for_missing_abstracts(papers))
    return dedupe_literature_papers_prefer_readable(enriched)


def fetch_semantic_scholar_details_for_missing_abstracts(papers: list[Any]) -> list[Any]:
    identifiers: list[str] = []
    limit = int(os.environ.get("WARA_PHASE1_ABSTRACT_ENRICH_LIMIT", "16") or 16)
    for paper in papers:
        if paper_has_readable_content(paper):
            continue
        doi = str(getattr(paper, "doi", "") or "").strip()
        arxiv_id = str(getattr(paper, "arxiv_id", "") or "").strip()
        if doi:
            identifiers.append(f"DOI:{doi}")
        elif arxiv_id:
            identifiers.append(f"ARXIV:{arxiv_id}")
        if len(identifiers) >= limit:
            break
    if not identifiers:
        return []
    try:
        from wara_core.literature.semantic_scholar import batch_fetch_papers

        api_key = os.environ.get("S2_API_KEY", "") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        return batch_fetch_papers(identifiers, api_key=api_key)
    except Exception:  # noqa: BLE001
        return []


def fetch_arxiv_details_for_missing_abstracts(papers: list[Any]) -> list[Any]:
    results: list[Any] = []
    limit = int(os.environ.get("WARA_PHASE1_ARXIV_ABSTRACT_ENRICH_LIMIT", "6") or 6)
    for paper in papers:
        if len(results) >= limit:
            break
        if paper_has_readable_content(paper):
            continue
        arxiv_id = str(getattr(paper, "arxiv_id", "") or "").strip()
        if not arxiv_id:
            continue
        try:
            from wara_core.literature.arxiv_client import get_paper_by_id

            enriched = get_paper_by_id(arxiv_id)
            if enriched is not None:
                results.append(enriched)
        except Exception:  # noqa: BLE001
            continue
    return results


_LITERATURE_SIGNAL_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "system_setting": {
        "cell-free massive MIMO": ("cell-free", "cell free"),
        "massive MIMO": ("massive mimo", "large-scale antenna", "large scale antenna"),
        "RIS-assisted network": ("ris", "reconfigurable intelligent surface", "intelligent reflecting surface"),
        "ISAC": ("isac", "integrated sensing", "sensing and communication"),
        "SWIPT/WPT": ("swipt", "wireless power transfer", "energy harvesting", "wireless-powered"),
        "UAV network": ("uav", "unmanned aerial vehicle", "drone"),
        "near-field network": ("near-field", "near field", "xl-mimo", "extremely large"),
    },
    "optimization_variables": {
        "beamforming/precoding": ("beamforming", "precoding", "beamformer", "transmit beam"),
        "power allocation": ("power allocation", "power control", "power loading", "transmit power"),
        "AP-user association": ("association", "clustering", "user-centric", "serving link"),
        "RIS phase shifts": ("phase shift", "reflection coefficient", "passive beamforming"),
        "trajectory/deployment": ("trajectory", "placement", "deployment", "positioning"),
        "scheduling/resource allocation": ("scheduling", "resource allocation", "bandwidth allocation"),
    },
    "objectives_metrics": {
        "sum rate/spectral efficiency": ("sum rate", "spectral efficiency", "se", "throughput"),
        "max-min fairness": ("max-min", "minimum rate", "fairness", "worst-user"),
        "energy efficiency": ("energy efficiency", "energy-efficient"),
        "harvested energy/power": ("harvested energy", "harvested power", "dc power"),
        "sensing accuracy/CRB": ("crb", "cramer-rao", "sensing accuracy", "estimation error"),
        "outage/reliability": ("outage", "reliability", "robust"),
        "latency": ("latency", "delay"),
    },
    "constraint_structure": {
        "power budget": ("power budget", "transmit-power", "maximum power", "power constraint"),
        "SINR/rate constraint": ("sinr", "rate constraint", "quality-of-service", "qos"),
        "fronthaul capacity": ("fronthaul", "backhaul", "capacity-limited"),
        "binary association": ("binary", "association", "integer", "mixed-integer"),
        "imperfect CSI/robustness": ("imperfect csi", "channel uncertainty", "robust"),
        "rank/semidefinite constraint": ("rank", "semidefinite", "sdr"),
        "energy-harvesting constraint": ("energy harvesting", "harvested energy", "power splitting"),
    },
    "solution_route": {
        "SCA": ("successive convex", "sca", "majorization", "minorization"),
        "SDR/SDP": ("semidefinite relaxation", "sdr", "sdp", "semidefinite programming"),
        "SOCP/MISOCP": ("second-order cone", "socp", "misocp", "conic"),
        "WMMSE": ("wmmse", "weighted minimum mean square error"),
        "alternating optimization": ("alternating optimization", "block coordinate", "alternating direction"),
        "bisection": ("bisection", "target rate", "feasibility test"),
        "mixed-integer optimization": ("mixed-integer", "integer programming", "branch-and-bound"),
        "robust approximation": ("robust optimization", "chance constraint", "uncertainty set"),
    },
}


def build_literature_cards(
    papers: list[Any],
    seminal_matches: list[dict[str, Any]],
    *,
    limit: int = 16,
    artifact_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build compact, traceable cards from abstract/PDF-backed records."""

    cards: list[dict[str, Any]] = []
    reading_records: list[dict[str, Any]] = []
    pdf_budget = int(os.environ.get("WARA_PHASE1_PDF_EXTRACTION_LIMIT", "4") or 4)
    pdf_attempts = 0
    for paper in papers[:limit]:
        title = str(getattr(paper, "title", "") or "").strip()
        if not title:
            continue
        abstract = str(getattr(paper, "abstract", "") or "").strip()
        pdf_record: dict[str, Any] = {
            "title": title,
            "status": "not_attempted",
            "reason": "pdf_extraction_disabled_or_no_budget",
        }
        if phase1_pdf_extraction_enabled() and pdf_attempts < pdf_budget:
            pdf_url = candidate_pdf_url_for_paper(paper)
            if pdf_url:
                pdf_attempts += 1
                pdf_record = extract_pdf_grounding(
                    pdf_url,
                    title=title,
                    artifact_dir=artifact_dir,
                )
        reading_records.append(pdf_record)
        pdf_text = str(pdf_record.get("text_snippet") or "").strip()
        pdf_abstract = str(pdf_record.get("abstract") or "").strip()
        card_abstract = abstract if len(abstract) >= 80 else pdf_abstract
        if len(card_abstract) < 80 and not pdf_text:
            continue
        text = " ".join(
            [
                title,
                card_abstract,
                pdf_text[:1200],
                str(getattr(paper, "venue", "") or ""),
                str(getattr(paper, "source", "") or ""),
            ]
        )
        card_id = f"paper-{len(cards) + 1}"
        cards.append(
            {
                "card_id": card_id,
                "title": title,
                "cite_key": str(getattr(paper, "cite_key", "") or ""),
                "year": int(getattr(paper, "year", 0) or 0),
                "venue": str(getattr(paper, "venue", "") or ""),
                "source": str(getattr(paper, "source", "") or ""),
                "doi": str(getattr(paper, "doi", "") or ""),
                "arxiv_id": str(getattr(paper, "arxiv_id", "") or ""),
                "url": str(getattr(paper, "url", "") or ""),
                "evidence_level": "pdf_text" if pdf_text else "abstract",
                "abstract_snippet": card_abstract[:900],
                "pdf_text_snippet": pdf_text[:1800],
                "pdf_extraction_status": pdf_record.get("status", "not_attempted"),
                "signals": extract_literature_signals(text),
            }
        )

    if not cards and seminal_matches:
        reading_records.append(
            {
                "status": "seed_reference_only",
                "reason": "local seed references support citation grounding but are not used as abstract-backed gap cards",
                "seed_reference_count": len(seminal_matches),
            }
        )
    return cards, reading_records


def phase1_pdf_extraction_enabled() -> bool:
    return os.environ.get("WARA_PHASE1_PDF_EXTRACTION", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
    }


def candidate_pdf_url_for_paper(paper: Any) -> str:
    arxiv_id = normalize_arxiv_id(str(getattr(paper, "arxiv_id", "") or ""))
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    url = str(getattr(paper, "url", "") or "").strip()
    if re.match(r"^https://arxiv\.org/abs/", url):
        return re.sub(r"/abs/", "/pdf/", url).rstrip("/") + ".pdf"
    if re.match(r"^https://[^\s]+\.pdf(?:\?.*)?$", url, flags=re.IGNORECASE):
        return url
    return ""


def normalize_arxiv_id(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", raw)
    raw = raw.removesuffix(".pdf")
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", raw)
    return match.group(1) if match else raw


def extract_pdf_grounding(
    pdf_url: str,
    *,
    title: str,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "title": title,
        "pdf_url": pdf_url,
        "status": "failed",
        "abstract": "",
        "text_snippet": "",
    }
    try:
        if not pdf_url.startswith("https://"):
            record["reason"] = "only_https_pdf_urls_are_allowed"
            return record
        req = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "WARA/1.0 PDF grounding"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read(int(os.environ.get("WARA_PHASE1_PDF_MAX_BYTES", "8000000") or 8000000))
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(data)
            tmp_name = handle.name
        try:
            text = extract_pdf_text(Path(tmp_name))
        finally:
            Path(tmp_name).unlink(missing_ok=True)
        text = compact_text(text)
        if not text:
            record["reason"] = "pdf_text_empty"
            return record
        abstract = extract_abstract_from_text(text)
        record.update(
            {
                "status": "ok",
                "abstract": abstract[:1200],
                "text_snippet": text[:4000],
                "chars": len(text),
            }
        )
        if artifact_dir is not None and os.environ.get("WARA_PHASE1_SAVE_PDF_TEXT", "").strip().lower() in {"1", "true", "yes", "on"}:
            text_dir = artifact_dir / "paper_text"
            text_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.lower()).strip("_")[:70] or "paper"
            write_text(text_dir / f"{safe_name}.txt", text[:8000])
            record["saved_text"] = str(text_dir / f"{safe_name}.txt")
    except Exception as exc:  # noqa: BLE001
        record["reason"] = f"{type(exc).__name__}: {exc}"
    return record


def extract_pdf_text(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyPDF2 is not installed") from exc

    reader = PdfReader(str(path))
    max_pages = int(os.environ.get("WARA_PHASE1_PDF_MAX_PAGES", "5") or 5)
    chunks: list[str] = []
    for index, page in enumerate(reader.pages):
        if index >= max_pages:
            break
        try:
            chunks.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    return "\n".join(chunks)


def extract_abstract_from_text(text: str) -> str:
    match = re.search(
        r"(?:^|\n)\s*abstract\s*[:\n]\s*(.*?)(?=\n\s*(?:\d+\.?\s*)?(?:introduction|1\s+introduction)\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return compact_text(match.group(1))
    match = re.search(r"\babstract\b\s*[:\n]\s*(.{200,1200})", text, flags=re.IGNORECASE | re.DOTALL)
    return compact_text(match.group(1)) if match else ""


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_literature_signals(text: str) -> dict[str, list[str]]:
    lower = str(text or "").lower()
    signals: dict[str, list[str]] = {}
    for category, patterns in _LITERATURE_SIGNAL_PATTERNS.items():
        hits: list[str] = []
        for label, terms in patterns.items():
            if any(term in lower for term in terms):
                hits.append(label)
        signals[category] = hits
    return signals


def build_literature_gap_signals(
    *,
    topic: str,
    research_payload: dict[str, Any],
    selected: dict[str, Any],
    literature_cards: list[dict[str, Any]],
    evidence_needed: list[Any],
) -> list[dict[str, Any]]:
    """Infer conservative gap signals from the research frame and literature cards."""

    observed = aggregate_literature_signals(literature_cards)
    frame_text = build_positive_gap_signal_text(topic, research_payload, selected)
    expected = extract_literature_signals(frame_text)
    signals: list[dict[str, Any]] = []

    for statement, required_terms in detect_supported_coupling_signals(literature_cards):
        supporting_cards = cards_matching_terms(literature_cards, required_terms)
        if supporting_cards:
            signals.append(
                {
                    "signal_id": f"supported-{len(signals) + 1}",
                    "type": "supported_coupling",
                    "statement": statement,
                    "supporting_cards": supporting_cards,
                    "use_policy": "Use as a candidate gap mechanism, not as a novelty proof.",
                }
            )

    for category, expected_values in expected.items():
        missing = [value for value in expected_values if value not in observed.get(category, {})]
        if missing:
            signals.append(
                {
                    "signal_id": f"weak-{category}",
                    "type": "weak_or_missing_literature_signal",
                    "statement": f"The research frame needs {category.replace('_', ' ')} signals that are weakly represented in retrieved records: {', '.join(missing)}.",
                    "supporting_cards": [],
                    "use_policy": "Treat as an evidence need or risk unless later references support it.",
                }
            )

    for question in dedupe_texts(evidence_needed)[:5]:
        signals.append(
            {
                "signal_id": f"need-{len(signals) + 1}",
                "type": "evidence_need",
                "statement": question,
                "supporting_cards": [],
                "use_policy": "Use to guide query expansion and candidate risk, not to claim prior-art absence.",
            }
        )

    if not signals:
        signals.append(
            {
                "signal_id": "fallback-1",
                "type": "limited_grounding",
                "statement": "Retrieved records provide reference support but no strong mechanism-specific gap signal; Phase 1.3 should keep novelty claims provisional.",
                "supporting_cards": [],
                "use_policy": "Select only a conservative, Phase-2-feasible optimization problem.",
            }
        )
    return signals[:12]


def build_positive_gap_signal_text(topic: str, research_payload: dict[str, Any], selected: dict[str, Any]) -> str:
    """Use only positive research intent when detecting expected literature signals."""

    research = dict(research_payload.get("research_object") or {})
    system = dict(research_payload.get("wireless_system_seed") or {})
    hypothesis = dict(research_payload.get("mechanism_hypothesis") or {})
    selected = dict(selected or {})
    positive_payload = {
        "topic": topic,
        "research_object": select_dict_keys(
            research,
            (
                "research_question",
                "physical_mechanism",
                "decision_layer",
                "performance_gap",
                "expected_research_gain",
            ),
        ),
        "wireless_system_seed": select_dict_keys(
            system,
            (
                "nodes",
                "channel_model_seed",
                "controls",
                "parameters",
                "derived_quantities",
                "primary_kpis",
                "constraints_seed",
            ),
        ),
        "mechanism_hypothesis": select_dict_keys(
            hypothesis,
            (
                "why_gain_may_exist",
                "operating_regimes",
                "evidence_needed",
            ),
        ),
        "selected_candidate": select_dict_keys(
            selected,
            (
                "title",
                "problem_statement",
                "wireless_scenario",
                "research_angle",
                "controls",
                "parameters",
                "derived_quantities",
                "objective",
                "constraints",
                "expected_research_gain",
                "validation_metrics",
                "theoretical_route",
                "algorithm_route",
                "mechanism_for_gain",
                "mechanism_interaction",
                "resource_coupling_change",
                "new_coupling_or_tradeoff",
                "performance_bottleneck_addressed",
                "optimization_gap",
                "objective_constraint_structure",
                "evidence_alignment",
            ),
        ),
    }
    return json.dumps(positive_payload, ensure_ascii=False)


def select_dict_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if payload.get(key) not in (None, "", [], {})}


def aggregate_literature_signals(cards: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for card in cards:
        signals = dict(card.get("signals") or {})
        for category, values in signals.items():
            bucket = counts.setdefault(category, {})
            for value in coerce_list(values):
                label = str(value or "").strip()
                if label:
                    bucket[label] = bucket.get(label, 0) + 1
    return counts


def detect_supported_coupling_signals(cards: list[dict[str, Any]]) -> list[tuple[str, tuple[str, ...]]]:
    blob = " ".join(
        " ".join(
            [
                str(card.get("title") or ""),
                str(card.get("abstract_snippet") or ""),
                json.dumps(card.get("signals") or {}, ensure_ascii=False),
            ]
        ).lower()
        for card in cards
    )
    candidates = [
        (
            "Rate-dependent fronthaul or backhaul load can couple AP-user association with power/rate allocation.",
            ("fronthaul", "association"),
        ),
        (
            "Binary serving-link or clustering decisions can make wireless resource allocation mixed discrete-continuous.",
            ("association", "power"),
        ),
        (
            "SINR or QoS constraints can couple beamforming/power variables across users through interference.",
            ("sinr", "beamforming"),
        ),
        (
            "Imperfect CSI or uncertainty can turn the selected wireless objective into a robust feasibility or performance problem.",
            ("imperfect csi", "robust"),
        ),
        (
            "Energy-harvesting or wireless-power requirements can create a rate-energy tradeoff in resource allocation.",
            ("energy harvesting", "power allocation"),
        ),
        (
            "ISAC metrics can couple communication service and sensing accuracy through shared waveform or beamforming variables.",
            ("integrated sensing", "beamforming"),
        ),
    ]
    return [(statement, terms) for statement, terms in candidates if all(term in blob for term in terms)]


def cards_matching_terms(cards: list[dict[str, Any]], terms: tuple[str, ...]) -> list[str]:
    matched: list[str] = []
    for card in cards:
        text = " ".join(
            [
                str(card.get("title") or ""),
                str(card.get("abstract_snippet") or ""),
                json.dumps(card.get("signals") or {}, ensure_ascii=False),
            ]
        ).lower()
        if all(term in text for term in terms):
            card_id = str(card.get("card_id") or "").strip()
            if card_id:
                matched.append(card_id)
    return matched[:5]


def summarize_literature_cards(cards: list[dict[str, Any]], gap_signals: list[dict[str, Any]]) -> dict[str, Any]:
    levels: dict[str, int] = {}
    for card in cards:
        level = str(card.get("evidence_level") or "unknown")
        levels[level] = levels.get(level, 0) + 1
    return {
        "literature_card_count": len(cards),
        "evidence_levels": levels,
        "abstract_or_pdf_backed_cards": sum(
            1 for card in cards if card.get("evidence_level") in {"abstract", "pdf_text"}
        ),
        "gap_signal_count": len(gap_signals),
        "grounding_policy": "Gap signals are built from abstract-backed or PDF-text-backed literature cards. Metadata-only records may support references but cannot by themselves ground a research gap.",
    }


def build_literature_grounding_limitations(cards: list[dict[str, Any]]) -> list[str]:
    if not cards:
        return ["No abstract-backed or PDF-text-backed literature cards were built; the evidence gate should block gap selection unless metadata fallback is explicitly allowed."]
    abstract_backed = sum(1 for card in cards if card.get("evidence_level") == "abstract")
    pdf_backed = sum(1 for card in cards if card.get("evidence_level") == "pdf_text")
    limitations: list[str] = []
    if not pdf_backed:
        limitations.append("No PDF extraction succeeded; WARA falls back to abstract-backed reading for gap grounding.")
    if not abstract_backed and pdf_backed:
        limitations.append("PDF text is available but source abstracts were not available for these cards.")
    limitations.append("Literature cards ground problem selection but do not prove novelty, priority, or final citation correctness.")
    return limitations


def paper_refs_from_search(papers: list[Any]) -> list[dict[str, Any]]:
    refs = []
    for paper in papers:
        refs.append(
            {
                "title": getattr(paper, "title", ""),
                "year": getattr(paper, "year", 0),
                "venue": getattr(paper, "venue", ""),
                "citation_count": getattr(paper, "citation_count", 0),
                "doi": getattr(paper, "doi", ""),
                "arxiv_id": getattr(paper, "arxiv_id", ""),
                "url": getattr(paper, "url", ""),
                "source": getattr(paper, "source", ""),
                "cite_key": getattr(paper, "cite_key", ""),
            }
        )
    return refs


def sanitize_bibtex_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = " and ".join(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text


def seminal_refs_from_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        refs.append(
            {
                "title": match.get("title", ""),
                "year": match.get("year", ""),
                "venue": match.get("venue", ""),
                "citation_count": match.get("citation_count", 0),
                "doi": match.get("doi", ""),
                "arxiv_id": match.get("arxiv_id", ""),
                "url": match.get("url", ""),
                "source": "seminal_library",
                "cite_key": match.get("cite_key") or match.get("bib_key", ""),
            }
        )
    return refs


def seminal_matches_to_bibtex(matches: list[dict[str, Any]]) -> str:
    entries: list[str] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        key = sanitize_bibtex_value(match.get("cite_key") or match.get("bib_key", ""))
        title = sanitize_bibtex_value(match.get("title", ""))
        authors = sanitize_bibtex_value(match.get("authors", ""))
        venue = sanitize_bibtex_value(match.get("venue", ""))
        year = sanitize_bibtex_value(match.get("year", ""))
        if not key or not title or not venue:
            continue
        entry_type = "inproceedings" if re.search(r"\bproc\.|conference|workshop|symposium", venue, flags=re.I) else "article"
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        lines = [
            f"@{entry_type}{{{key},",
            f"  title = {{{title}}},",
        ]
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        lines.append(f"  {venue_field} = {{{venue}}},")
        doi = sanitize_bibtex_value(match.get("doi", ""))
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        url = sanitize_bibtex_value(match.get("url", ""))
        if url:
            lines.append(f"  url = {{{url}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries).strip()


def combine_bibtex_blocks(*blocks: str) -> str:
    entries: list[str] = []
    seen_keys: set[str] = set()
    for block in blocks:
        for raw_entry in re.split(r"\n\s*\n(?=@)", str(block or "").strip()):
            entry = raw_entry.strip()
            if not entry:
                continue
            match = re.search(r"@\w+\s*\{\s*([^,\s]+)", entry)
            key = match.group(1).strip().lower() if match else entry[:80].lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            entries.append(entry)
    return ("\n\n".join(entries).strip() + "\n") if entries else ""


def count_bibtex_entries(text: str) -> int:
    return len(re.findall(r"@\w+\s*\{", str(text or "")))


def dedupe_reference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record.get("cite_key") or record.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def dedupe_structured_records(records: list[Any], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        key = ""
        for field in key_fields:
            value = str(record.get(field) or "").strip().lower()
            if value:
                key = f"{field}:{value}"
                break
        if not key:
            key = json.dumps(record, sort_keys=True, ensure_ascii=False)[:160].lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def dedupe_literature_papers(papers: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for paper in papers:
        title = re.sub(r"\s+", " ", str(getattr(paper, "title", "") or "").strip().lower())
        doi = str(getattr(paper, "doi", "") or "").strip().lower()
        arxiv_id = str(getattr(paper, "arxiv_id", "") or "").strip().lower()
        key = doi or (f"arxiv:{arxiv_id}" if arxiv_id else title)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(paper)
    return result


def dedupe_literature_papers_prefer_readable(papers: list[Any]) -> list[Any]:
    """Deduplicate papers while keeping the most useful record for gap grounding."""

    best_by_key: dict[str, tuple[int, int, Any]] = {}
    order_by_key: dict[str, int] = {}
    for index, paper in enumerate(papers):
        key = literature_paper_key(paper)
        if not key:
            continue
        order_by_key.setdefault(key, index)
        score = literature_paper_reading_score(paper)
        current = best_by_key.get(key)
        if current is None or score > current[0]:
            best_by_key[key] = (score, index, paper)

    selected = [
        (order_by_key.get(key, index), score, paper)
        for key, (score, index, paper) in best_by_key.items()
    ]
    selected.sort(
        key=lambda item: (
            0 if paper_has_readable_content(item[2]) else 1,
            -item[1],
            item[0],
        )
    )
    return [paper for _, _, paper in selected]


def literature_paper_key(paper: Any) -> str:
    title = re.sub(r"\s+", " ", str(getattr(paper, "title", "") or "").strip().lower())
    doi = str(getattr(paper, "doi", "") or "").strip().lower()
    arxiv_id = normalize_arxiv_id(str(getattr(paper, "arxiv_id", "") or "")).lower()
    return doi or (f"arxiv:{arxiv_id}" if arxiv_id else title)


def literature_paper_reading_score(paper: Any) -> int:
    score = 0
    abstract = str(getattr(paper, "abstract", "") or "").strip()
    if len(abstract) >= 80:
        score += 1_000_000 + min(len(abstract), 5000)
    if candidate_pdf_url_for_paper(paper):
        score += 100_000
    source = str(getattr(paper, "source", "") or "").strip().lower()
    if source in {"semantic_scholar", "openalex", "arxiv", "ieee_xplore"}:
        score += 10_000
    try:
        score += min(int(getattr(paper, "citation_count", 0) or 0), 5000)
    except (TypeError, ValueError):
        pass
    try:
        score += min(max(int(getattr(paper, "year", 0) or 0) - 2000, 0), 40)
    except (TypeError, ValueError):
        pass
    return score


def evidence_pack_reference_count(evidence_pack: dict[str, Any]) -> int:
    return max(
        len(coerce_list(evidence_pack.get("references"))),
        count_bibtex_entries(str(evidence_pack.get("references_bib") or "")),
    )


def validate_evidence_pack_reference_contract(evidence_pack: dict[str, Any], *, minimum: int | None = None) -> dict[str, Any]:
    if minimum is None:
        minimum = int(os.environ.get("WARA_PHASE1_REFERENCE_MIN", "12") or 12)
    count = evidence_pack_reference_count(evidence_pack)
    report = {
        "ok": count >= minimum,
        "minimum_reference_target": minimum,
        "reference_count": count,
        "search_mode": evidence_pack.get("search_mode", ""),
        "literature_sources": evidence_pack.get("literature_sources", []),
        "source_status": evidence_pack.get("source_status", {}),
    }
    if not report["ok"]:
        raise ValueError(
            "Phase 1 reference bank contract failed: "
            f"{count} references < hard target {minimum}. "
            "This is a LiteratureAgent/handoff error; do not let Phase 2 repair it by adding references."
        )
    return report


def validate_literature_grounding_contract(evidence_pack: dict[str, Any], *, minimum: int | None = None) -> dict[str, Any]:
    search_enabled = evidence_pack.get("search_mode") == "external_literature_search"
    if minimum is None:
        minimum = phase1_minimum_abstract_cards(search_enabled=search_enabled)
    cards = [dict(item) for item in coerce_list(evidence_pack.get("literature_cards")) if isinstance(item, dict)]
    readable = [
        card
        for card in cards
        if card.get("evidence_level") in {"abstract", "pdf_text"}
        and len(str(card.get("abstract_snippet") or card.get("pdf_text_snippet") or "").strip()) >= 80
    ]
    report = {
        "ok": len(readable) >= minimum,
        "minimum_abstract_or_pdf_cards": minimum,
        "abstract_or_pdf_card_count": len(readable),
        "paper_reading_records": len(coerce_list(evidence_pack.get("paper_reading_records"))),
        "search_mode": evidence_pack.get("search_mode", ""),
    }
    if not report["ok"] and os.environ.get("WARA_PHASE1_ALLOW_METADATA_GAP_FALLBACK", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise ValueError(
            "Phase 1 literature grounding contract failed: "
            f"{len(readable)} abstract/PDF-backed literature cards < required {minimum}. "
            "WARA cannot identify a reliable research gap from metadata-only references."
        )
    return report


def dedupe_texts(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def normalize_topic_literature(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_title": "",
        "references": evidence_pack.get("references", []),
        "seminal_matches": evidence_pack.get("seminal_matches", []),
        "search_queries": evidence_pack.get("reference_search_queries", []),
        "literature_sources": evidence_pack.get("literature_sources", []),
        "source_status": evidence_pack.get("source_status", {}),
        "evidence_needed": evidence_pack.get("evidence_needed", []),
        "literature_cards": evidence_pack.get("literature_cards", []),
        "paper_reading_records": evidence_pack.get("paper_reading_records", []),
        "gap_signals": evidence_pack.get("gap_signals", []),
        "literature_evidence_summary": evidence_pack.get("literature_evidence_summary", {}),
        "literature_grounding_limitations": evidence_pack.get("literature_grounding_limitations", []),
        "citation_policy": evidence_pack.get("citation_policy", ""),
        "search_mode": evidence_pack.get("search_mode", ""),
    }


def render_goal_markdown(topic: str) -> str:
    return f"""# WARA Phase 1 Topic Intake

## User Topic
{topic}

## Phase 1 Mode
WARA-native 4-phase controller with artifact-mediated ScoutAgent, LiteratureAgent, and controller handoff export.

## Output Contract
The run must produce `phase1_handoff.json` for Phase 2. It must not rely on legacy literature-phase handoff semantics.
"""


def render_scope_markdown(topic: str, payload: dict[str, Any]) -> str:
    profile = dict(payload.get("scope_contract") or payload.get("topic_profile") or {})
    ontology = dict(payload.get("wireless_ontology") or {})
    taxonomy = dict(payload.get("taxonomy_plan") or {})
    return f"""# Scope Contract

## User Topic
{topic}

## Domain
{profile.get("domain", "wireless communications")}

## Preserved Mechanisms
{format_list(profile.get("preserved_mechanisms"))}

## Wireless Ontology Summary
{ontology.get("summary", "TBD")}

## Missing Taxonomy Layers
{format_list(taxonomy.get("missing_layers"))}

## Scope Boundary
{profile.get("scope_boundary", "TBD")}

## Phase-2 Risks
{format_list(profile.get("phase2_risks"))}
"""


def render_research_object_markdown(payload: dict[str, Any]) -> str:
    research = dict(payload.get("research_object") or {})
    system = dict(payload.get("wireless_system_seed") or {})
    hypothesis = dict(payload.get("mechanism_hypothesis") or {})
    readiness = dict(payload.get("phase2_readiness") or {})
    return f"""# Research Object

## Research Question
{research.get("research_question", "TBD")}

## Physical Mechanism
{research.get("physical_mechanism", "TBD")}

## Expected Research Gain
{format_list(research.get("expected_research_gain"))}

## Controls
{format_list(system.get("controls"))}

## Primary KPIs
{format_list(system.get("primary_kpis"))}

## Why Gain May Exist
{hypothesis.get("why_gain_may_exist", "TBD")}

## Phase-2 Needs
{format_list(readiness.get("formulation_needs"))}
"""


def render_candidates_markdown(candidates: list[Any]) -> str:
    lines = ["# Candidate Directions", ""]
    for idx, item in enumerate(candidates, start=1):
        cand = dict(item or {})
        lines.extend(
            [
                f"## Candidate {idx}: {cand.get('title', cand.get('id', 'Untitled'))}",
                "",
                f"**Problem.** {cand.get('problem_statement', '')}",
                "",
                f"**Objective.** {cand.get('objective', '')}",
                "",
                "**Expected research gain.**",
                format_list(cand.get("expected_research_gain")),
                "",
                "**Validation metrics.**",
                format_list(cand.get("validation_metrics")),
                "",
                "**Kill criteria.**",
                format_list(cand.get("kill_criteria")),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def render_evidence_pack_markdown(evidence_pack: dict[str, Any]) -> str:
    summary = dict(evidence_pack.get("literature_evidence_summary") or {})
    summary_lines = [
        f"{key}: {value}"
        for key, value in summary.items()
        if key != "grounding_policy"
    ]
    if summary.get("grounding_policy"):
        summary_lines.append(str(summary.get("grounding_policy")))
    gap_lines = [
        f"{item.get('signal_id', 'signal')}: {item.get('statement', '')}"
        for item in coerce_list(evidence_pack.get("gap_signals"))
        if isinstance(item, dict)
    ]
    card_lines = [
        f"{item.get('card_id', 'card')}: {item.get('title', '')} [{item.get('evidence_level', 'unknown')}]"
        for item in coerce_list(evidence_pack.get("literature_cards"))
        if isinstance(item, dict)
    ]
    reading_lines = [
        f"{item.get('status', 'unknown')}: {item.get('title', item.get('pdf_url', 'paper'))}"
        for item in coerce_list(evidence_pack.get("paper_reading_records"))
        if isinstance(item, dict)
    ]
    return f"""# Evidence Pack

## Search Mode
{evidence_pack.get("search_mode", "local")}

## Literature Sources
{format_list(evidence_pack.get("literature_sources"))}

## Source Status
{format_list([f"{key}: {value}" for key, value in dict(evidence_pack.get("source_status") or {}).items()])}

## Literature Questions
{format_list(evidence_pack.get("literature_questions"))}

## Evidence Needed
{format_list(evidence_pack.get("evidence_needed"))}

## Reference Search Queries
{format_list(evidence_pack.get("reference_search_queries"))}

## Seminal Matches
{format_list([item.get("title", "") if isinstance(item, dict) else item for item in coerce_list(evidence_pack.get("seminal_matches"))])}

## Retrieved References
{format_list([item.get("title", "") if isinstance(item, dict) else item for item in coerce_list(evidence_pack.get("references"))])}

## Literature Evidence Summary
{format_list(summary_lines)}

## Gap Signals
{format_list(gap_lines)}

## Literature Cards
{format_list(card_lines)}

## Paper Reading Records
{format_list(reading_lines)}

## Grounding Limitations
{format_list(evidence_pack.get("literature_grounding_limitations"))}

## Citation Policy
{evidence_pack.get("citation_policy", "Do not fabricate citations.")}
"""


def render_hypotheses_markdown(payload: dict[str, Any]) -> str:
    selected = dict(payload.get("selected_candidate") or {})
    problem = dict(payload.get("problem_contract_seed") or {})
    novelty = dict(payload.get("novelty_contract") or {})
    proof = dict(payload.get("proof_contract") or {})
    validation = dict(payload.get("validation_contract") or {})
    return f"""# Selected WARA Candidate

## Candidate 1: {selected.get("title", "TBD")}

**Problem statement**
{selected.get("problem_statement", "TBD")}

**Wireless scenario**
{selected.get("wireless_scenario", "TBD")}

**Objective**
{selected.get("objective") or problem.get("objective") or "TBD"}

**Declared variables**
{format_list(selected.get("variables") or problem.get("controls"))}

**Core constraints**
{format_list(selected.get("core_constraints") or problem.get("constraints"))}

**Claimed contribution**
{selected.get("claimed_contribution", "TBD")}

**Novelty delta vs prior art**
{selected.get("novelty_delta") or novelty.get("claim_boundary") or "TBD"}

**Theorem / proof target**
{format_list(proof.get("target_claims") or selected.get("theorem_or_algorithmic_claim"))}

**Tractability / reformulation path**
{selected.get("tractability_path") or selected.get("convexification_path") or proof.get("route") or "TBD"}

**Minimal validation plan**
{format_list(validation.get("figures"))}

**Kill criteria**
{format_list(payload.get("kill_criteria"))}
"""


def render_topic_score(payload: dict[str, Any], candidate_review: dict[str, Any]) -> dict[str, Any]:
    decision = dict(candidate_review.get("selection_decision") or {})
    try:
        score = float(decision.get("readiness_score_1_to_10") or 8.0)
    except (TypeError, ValueError):
        score = 8.0
    selected = dict(payload.get("selected_candidate") or {})
    return {
        "generated_at": utcnow_iso(),
        "recommended_title": selected.get("title", ""),
        "recommended_candidate": selected.get("title", ""),
        "verdict": "proceed",
        "overall_score": max(0.0, min(10.0, score)),
        "dimension_scores": {
            "novelty": max(0.0, min(10.0, score - 0.2)),
            "feasibility": max(0.0, min(10.0, score)),
            "wireless_fit": max(0.0, min(10.0, score + 0.2)),
            "scope_fit": max(0.0, min(10.0, score)),
            "validation_clarity": max(0.0, min(10.0, score - 0.1)),
        },
        "justification": "WARA-native Phase 1 readiness score from the selection decision; downstream novelty and citation checks remain required.",
    }


def render_tail_summary(run_dir: Path, payload: dict[str, Any], topic_score: dict[str, Any]) -> dict[str, Any]:
    selected = dict(payload.get("selected_candidate") or {})
    return {
        "source_run": str(run_dir),
        "selected_title": selected.get("title", ""),
        "topic": selected.get("title", ""),
        "overall_score": topic_score.get("overall_score"),
        "handoff_file": str(run_dir / "phase1-4" / "phase1_handoff.json"),
        "generated_at": utcnow_iso(),
        "phase1_design": "wara_native_4_phase_controller",
        "phase": "phase1",
    }


def write_run_summary(run_dir: Path, handoff_dir: Path, topic: str, selected_title: str, trace: list[dict[str, Any]]) -> None:
    write_text(
        run_dir / "pipeline_summary.json",
        dump_json(
            {
                "run_id": run_dir.name,
                "topic": topic,
                "selected_title": selected_title,
                "status": "completed",
        "phase1_design": "wara_native_4_phase_controller",
                "phase": "phase1",
                "handoff_dir": str(handoff_dir),
                "phase_step_count": 4,
                "agent_trace": trace,
                "generated_at": utcnow_iso(),
            }
        ),
    )


def phase1_step_dir(run_dir: Path, phase_num: int) -> Path:
    return run_dir / f"phase1-{phase_num}"


def write_phase_status(run_dir: Path, phase_num: int, artifacts: tuple[str, ...]) -> None:
    phase_dir = phase1_step_dir(run_dir, phase_num)
    write_text(
        phase_dir / "decision.json",
        dump_json(
            {
                "status": "done",
                "decision": "proceed",
                "phase_step": phase_num,
                "phase_id": PHASE_IDS.get(phase_num, f"phase1.{phase_num}"),
                "artifacts": list(artifacts),
                "generated_at": utcnow_iso(),
            }
        ),
    )


def write_phase_artifact(run_dir: Path, phase_num: int, name: str, content: str) -> None:
    write_text(phase1_step_dir(run_dir, phase_num) / name, content)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object found")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def format_list(value: Any) -> str:
    items = coerce_list(value)
    if not items:
        return "- TBD"
    return "\n".join(f"- {item}" for item in items if str(item).strip()) or "- TBD"


def format_inline_list(value: Any) -> str:
    items = [str(item).strip() for item in coerce_list(value) if str(item).strip()]
    return ", ".join(items)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_workspace_env() -> None:
    for env_path in (WORKSPACE_ROOT / ".env", PHASE1_ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
