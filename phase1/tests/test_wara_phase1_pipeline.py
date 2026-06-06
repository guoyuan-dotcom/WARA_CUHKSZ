from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


PHASE1_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PHASE1_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from wara_phase1_pipeline import (  # noqa: E402
    build_literature_cards,
    build_literature_gap_signals,
    build_direction_contract_prompt,
    build_research_frame_payload,
    build_research_object_prompt,
    build_wireless_scope_context,
    combine_bibtex_blocks,
    filter_relevant_literature_papers,
    build_literature_source_status,
    evidence_pack_reference_count,
    dedupe_literature_papers_prefer_readable,
    merge_evidence_packs,
    normalize_literature_source,
    resolve_phase1_literature_sources,
    run_wara_phase1,
    seminal_matches_to_bibtex,
    phase1_literature_search_enabled,
    validate_evidence_pack_reference_contract,
    validate_literature_grounding_contract,
    validate_handoff_payload,
)
from wara_core.domains.wireless_topic_taxonomy import build_wireless_topic_taxonomy_plan  # noqa: E402
from wara_core.literature.models import Paper  # noqa: E402


@dataclass
class FakeResponse:
    content: str
    model: str = "mock-model"
    total_tokens: int = 123


class FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def chat(self, messages, *, system=None, max_tokens=None, temperature=None, json_mode=False):
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_mode": json_mode,
            }
        )
        if not self.payloads:
            raise AssertionError("unexpected extra LLM call")
        return FakeResponse(json.dumps(self.payloads.pop(0)))


class WaraPhase1PipelineTest(unittest.TestCase):
    def test_seminal_matches_are_exported_to_phase2_bibtex(self) -> None:
        seminal = [
            {
                "bib_key": "ShiLiuXuZhang2014",
                "title": "Joint Transmit Beamforming and Receive Power Splitting for MISO SWIPT Systems",
                "authors": ["Q. Shi", "L. Liu", "W. Xu", "R. Zhang"],
                "venue": "IEEE Transactions on Wireless Communications",
                "year": 2014,
                "doi": "10.1109/TWC.2014.2328131",
            }
        ]

        bibtex = combine_bibtex_blocks("", seminal_matches_to_bibtex(seminal))

        self.assertIn("@article{ShiLiuXuZhang2014", bibtex)
        self.assertIn("Joint Transmit Beamforming", bibtex)
        self.assertIn("IEEE Transactions on Wireless Communications", bibtex)

    def test_selected_evidence_merge_preserves_base_references(self) -> None:
        base = {
            "reference_search_queries": ["near-field MIMO survey"],
            "references": [{"cite_key": "liu2024nearfield", "title": "Near-Field Communications: A Comprehensive Survey"}],
            "retrieved_references": [{"cite_key": "liu2024nearfield", "title": "Near-Field Communications: A Comprehensive Survey"}],
            "references_bib": "@article{liu2024nearfield,\n  title = {Near-Field Communications: A Comprehensive Survey},\n}\n",
        }
        selected = {
            "reference_search_queries": ["near-field ISACP beamforming"],
            "references": [{"cite_key": "anjum2025energyefficient", "title": "Energy-Efficient Near-Field Integrated Sensing and Communication"}],
            "retrieved_references": [{"cite_key": "anjum2025energyefficient", "title": "Energy-Efficient Near-Field Integrated Sensing and Communication"}],
            "references_bib": "@article{anjum2025energyefficient,\n  title = {Energy-Efficient Near-Field Integrated Sensing and Communication},\n}\n",
        }

        merged = merge_evidence_packs(base, selected)

        self.assertEqual(len(merged["references"]), 2)
        self.assertIn("liu2024nearfield", merged["references_bib"])
        self.assertIn("anjum2025energyefficient", merged["references_bib"])
        self.assertEqual(merged["evidence_merge_policy"], "base_topic_evidence_plus_selected_direction_evidence")

    def test_phase1_reference_contract_blocks_thin_handoff(self) -> None:
        evidence_pack = {
            "references": [{"cite_key": "OnlyOne2025", "title": "Only one reference"}],
            "references_bib": "@article{OnlyOne2025,\n  title={Only one reference},\n}\n",
        }

        with self.assertRaisesRegex(ValueError, "1 references < hard target 12"):
            validate_evidence_pack_reference_contract(evidence_pack)

        self.assertEqual(evidence_pack_reference_count(evidence_pack), 1)

    def test_phase1_literature_search_is_enabled_by_default(self) -> None:
        old_value = os.environ.get("WARA_PHASE1_LITERATURE_SEARCH")
        os.environ.pop("WARA_PHASE1_LITERATURE_SEARCH", None)
        try:
            self.assertTrue(phase1_literature_search_enabled())
            os.environ["WARA_PHASE1_LITERATURE_SEARCH"] = "0"
            self.assertFalse(phase1_literature_search_enabled())
            os.environ["WARA_PHASE1_LITERATURE_SEARCH"] = "false"
            self.assertFalse(phase1_literature_search_enabled())
            os.environ["WARA_PHASE1_LITERATURE_SEARCH"] = "1"
            self.assertTrue(phase1_literature_search_enabled())
        finally:
            if old_value is None:
                os.environ.pop("WARA_PHASE1_LITERATURE_SEARCH", None)
            else:
                os.environ["WARA_PHASE1_LITERATURE_SEARCH"] = old_value

    def test_literature_cards_require_abstract_or_pdf_text(self) -> None:
        papers = [
            Paper(
                paper_id="metadata-only",
                title="Metadata Only Cell-Free Massive MIMO Paper",
                year=2025,
                abstract="",
                venue="IEEE Wireless Communications Letters",
                source="crossref",
            ),
            Paper(
                paper_id="abstract-backed",
                title="Fronthaul-Aware User-Centric Cell-Free Massive MIMO Optimization",
                year=2025,
                abstract=(
                    "This paper studies fronthaul-aware user-centric cell-free massive MIMO optimization "
                    "where AP-user association, power loading, and rate constraints are jointly considered "
                    "under finite fronthaul capacity and fairness requirements."
                ),
                venue="IEEE Transactions on Wireless Communications",
                source="semantic_scholar",
            ),
        ]

        old_pdf = os.environ.get("WARA_PHASE1_PDF_EXTRACTION")
        try:
            os.environ["WARA_PHASE1_PDF_EXTRACTION"] = "0"
            cards, reading_records = build_literature_cards(papers, [])
        finally:
            if old_pdf is None:
                os.environ.pop("WARA_PHASE1_PDF_EXTRACTION", None)
            else:
                os.environ["WARA_PHASE1_PDF_EXTRACTION"] = old_pdf

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["evidence_level"], "abstract")
        self.assertIn("Fronthaul-Aware", cards[0]["title"])
        self.assertEqual(len(reading_records), 2)

    def test_literature_dedupe_prefers_readable_record(self) -> None:
        metadata_only = Paper(
            paper_id="crossref-1",
            title="Joint Power Control for Wireless Networks",
            year=2024,
            abstract="",
            citation_count=500,
            doi="10.1109/example",
            source="crossref",
        )
        abstract_backed = Paper(
            paper_id="s2-1",
            title="Joint Power Control for Wireless Networks",
            year=2024,
            abstract=(
                "This work studies joint power control for wireless networks with coupled interference, "
                "quality-of-service constraints, and optimization-based resource allocation. "
                "The abstract is intentionally long enough for grounding."
            ),
            citation_count=1,
            doi="10.1109/example",
            source="semantic_scholar",
        )

        deduped = dedupe_literature_papers_prefer_readable([metadata_only, abstract_backed])

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].paper_id, "s2-1")

    def test_literature_grounding_contract_blocks_metadata_only_gap(self) -> None:
        evidence_pack = {
            "search_mode": "external_literature_search",
            "literature_cards": [],
            "paper_reading_records": [{"status": "not_attempted", "title": "Metadata only"}],
        }

        with self.assertRaisesRegex(ValueError, "metadata-only references"):
            validate_literature_grounding_contract(evidence_pack, minimum=1)

    def test_gap_signal_extraction_ignores_negative_scope_fields(self) -> None:
        research_payload = {
            "research_object": {
                "research_question": "How should cell-free massive MIMO optimize AP-user association?",
                "physical_mechanism": "Distributed APs jointly serve users through coherent transmission.",
                "decision_layer": "AP-user association and power allocation.",
                "performance_gap": "Fronthaul load couples clustering and rate allocation.",
                "expected_research_gain": ["fairer admitted rates"],
                "non_goals": ["Do not add UAV, near-field, RIS, SWIPT, ISAC, or NOMA."],
            },
            "wireless_system_seed": {
                "controls": ["association", "power"],
                "primary_kpis": ["minimum rate"],
                "constraints_seed": ["fronthaul capacity", "power budget"],
            },
            "mechanism_hypothesis": {
                "why_gain_may_exist": "Rate-dependent fronthaul load changes serving-link value.",
                "evidence_needed": ["fronthaul-aware user-centric cell-free massive MIMO"],
            },
        }
        cards = [
            {
                "card_id": "paper-1",
                "title": "Fronthaul-aware cell-free massive MIMO",
                "abstract_snippet": "Cell-free massive MIMO with fronthaul-aware association and power allocation.",
                "signals": {
                    "system_setting": ["cell-free massive MIMO"],
                    "optimization_variables": ["AP-user association", "power allocation"],
                    "constraint_structure": ["fronthaul capacity", "power budget"],
                },
            }
        ]

        signals = build_literature_gap_signals(
            topic="cell-free massive MIMO",
            research_payload=research_payload,
            selected={},
            literature_cards=cards,
            evidence_needed=[],
        )
        weak_text = " ".join(item["statement"] for item in signals if item.get("type") == "weak_or_missing_literature_signal")

        self.assertNotIn("UAV network", weak_text)
        self.assertNotIn("near-field network", weak_text)
        self.assertNotIn("RIS-assisted network", weak_text)

    def test_run_wara_phase1_writes_phase2_native_handoff(self) -> None:
        research_payload = {
            "topic_profile": {
                "domain": "wireless communications",
                "user_topic": "SWIPT nonlinear EH",
                "preserved_mechanisms": ["SWIPT", "nonlinear EH"],
                "forbidden_added_mechanisms": ["UAV", "RIS"],
                "scope_boundary": "single-cell WCL scope",
                "phase2_risks": ["novelty must be verified"],
            },
            "research_object": {
                "research_question": "How should nonlinear EH shape SWIPT resource allocation?",
                "physical_mechanism": "rectifier saturation changes the rate-energy tradeoff",
                "decision_layer": "beamforming and power splitting",
                "performance_gap": "linear EH can allocate power in the wrong regime",
                "expected_research_gain": ["rectifier-aware utility improves operating-regime decisions"],
                "non_goals": ["do not add UAV or RIS"],
            },
            "wireless_system_seed": {
                "nodes": ["BS", "SWIPT users"],
                "channel_model_seed": "flat fading",
                "csi_assumption_seed": "perfect CSI",
                "controls": ["w_k", "rho_k"],
                "parameters": ["P_max"],
                "derived_quantities": ["R_k", "E_k"],
                "primary_kpis": ["U"],
                "constraints_seed": ["sum_k ||w_k||^2 <= P_max"],
            },
            "mechanism_hypothesis": {
                "why_gain_may_exist": "nonlinear saturation rewards different PS choices",
                "operating_regimes": ["moderate P_max"],
                "failure_regimes": ["very low P_max"],
                "evidence_needed": ["nonlinear EH model reference"],
            },
            "phase2_readiness": {
                "formulation_needs": ["define U"],
                "theory_needs": ["SCA route"],
                "validation_needs": ["U curves"],
                "ambiguity_to_resolve": [],
            },
            "direction_constraints": {
                "must_preserve": ["SWIPT", "nonlinear EH"],
                "must_not_add": ["UAV", "RIS"],
                "allowed_abstractions": ["single-cell downlink"],
            },
        }
        direction_payload = {
            "candidate_directions": [
                {
                    "id": "c1",
                    "title": "Nonlinear SWIPT Utility Maximization",
                    "problem_statement": "Optimize nonlinear SWIPT utility.",
                    "wireless_scenario": "Single-cell SWIPT downlink.",
                    "preserved_topic_scope": "SWIPT nonlinear EH",
                    "controls": ["w_k", "rho_k"],
                    "parameters": ["P_max"],
                    "derived_quantities": ["R_k", "E_k"],
                    "objective": "maximize U",
                    "constraints": ["sum_k ||w_k||^2 <= P_max"],
                    "expected_research_gain": ["rectifier-aware utility improves operating-regime decisions"],
                    "theoretical_route": "SCA stationarity",
                    "algorithm_route": "successive convex approximation",
                    "validation_metrics": ["U", "sum R_k", "sum E_k"],
                    "figure_ideas": ["U versus P_max", "U versus b"],
                    "evidence_questions": ["Does nonlinear EH change the optimal PS operating regime?"],
                    "novelty_hypothesis": "nonlinear utility improves operating-regime decisions",
                    "risk_notes": ["prior nonlinear SWIPT may be close"],
                    "kill_criteria": ["no gain over linear EH"],
                }
            ],
            "selection_rubric": {"criteria": ["feasibility"], "weights": {"feasibility": 1}},
            "evidence_pack": {
                "literature_questions": ["Which nonlinear EH models are standard?"],
                "evidence_needed": ["nonlinear EH model reference"],
                "reference_search_queries": ["SWIPT nonlinear energy harvesting power splitting"],
                "citation_policy": "verify references downstream",
            },
            "selection_decision": {
                "selected_id": "c1",
                "selected_title": "Nonlinear SWIPT Utility Maximization",
                "rationale": "focused and testable",
                "rejected_ids": [],
                "readiness_score_1_to_10": 8.2,
            },
        }
        direction_contract_payload = {
            "candidate_directions": [
                {
                    "id": "c1",
                    "title": "Nonlinear SWIPT Utility Maximization",
                    "problem_statement": "Optimize nonlinear SWIPT utility.",
                    "wireless_scenario": "Single-cell SWIPT downlink.",
                    "research_angle": "rectifier-aware utility maximization",
                    "selected_extension_axis": "energy_or_hardware_response",
                    "concrete_mechanism": "nonlinear energy harvesting",
                    "mechanism_for_gain": "nonlinear saturation changes PS choices",
                    "mechanism_interaction": "nonlinear EH changes the marginal value of RF power delivered by SWIPT beams",
                    "resource_coupling_change": "power splitting and beam power must account for EH sensitivity and saturation instead of a fixed linear conversion",
                    "expected_kpi_gain": "higher weighted utility in nonlinear EH-sensitive operating regimes",
                    "operating_regime": "moderate P_max where EH receivers are near turn-on or saturation regions",
                    "tractability_risk": "SCA is needed for coupled SINR and nonlinear EH terms",
                    "combination_novelty": "SWIPT and nonlinear EH jointly create a nonconstant marginal harvested-energy value that is absent from linear SWIPT",
                    "why_not_keyword_stacking": "nonlinear EH changes the resource allocation law rather than only adding a label",
                    "new_coupling_or_tradeoff": "rate-energy tradeoff depends on EH turn-on and saturation regions",
                    "performance_bottleneck_addressed": "linear EH overestimates useful harvested energy after saturation",
                    "testable_gain_regime": "moderate input RF power where EH receivers transition between sensitivity and saturation",
                    "optimization_gap": "linear EH utility creates the wrong marginal objective for power splitting",
                    "optimization_novelty": "nonlinear EH adds a saturation-aware utility/constraint structure to SWIPT optimization",
                    "objective_constraint_structure": "maximize U with nonlinear harvested-energy terms and SINR constraints",
                    "algorithmic_route": "SCA with first-order surrogate updates for nonlinear EH and SINR coupling",
                    "evidence_alignment": "nonlinear EH models are used in SWIPT",
                    "phase2_risks": ["prior nonlinear SWIPT may be close"],
                    "kill_criteria": ["no gain over linear EH"],
                }
            ],
            "selection_decision": {
                "selected_id": "c1",
                "selected_title": "Nonlinear SWIPT Utility Maximization",
                "rationale": "focused and testable",
                "rejected_ids": [],
                "readiness_score_1_to_10": 8.2,
            },
            "selected_candidate": {
                "title": "Nonlinear SWIPT Utility Maximization",
                "problem_statement": "Optimize nonlinear SWIPT utility.",
                "wireless_scenario": "Single-cell SWIPT downlink.",
                "objective": "maximize U",
                "variables": ["w_k", "rho_k"],
                "core_constraints": ["sum_k ||w_k||^2 <= P_max"],
                "claimed_contribution": "rectifier-aware robust SCA design",
                "novelty_delta": "nonlinear EH-aware operating regime",
                "source_of_nonconvexity": "coupled SINR and nonlinear EH",
                "convexification_path": "SCA lower bounds",
                "theorem_or_algorithmic_claim": "monotone convergence to a stationary point",
                "expected_research_gain": "higher U in nonlinear EH-sensitive regimes",
            },
            "problem_contract_seed": {
                "controls": ["w_k", "rho_k"],
                "parameters": ["P_max"],
                "derived_quantities": ["R_k", "E_k"],
                "objective": "maximize U",
                "constraints": ["sum_k ||w_k||^2 <= P_max"],
                "assumptions": ["perfect CSI"],
                "primary_kpis": ["U"],
            },
            "novelty_contract": {
                "claim_boundary": "operating-regime gains under nonlinear EH",
                "prior_art_boundary": "existing nonlinear SWIPT formulations",
                "novelty_hypothesis": "nonlinear EH changes optimal PS",
                "optimization_novelty": "nonlinear EH changes the objective and feasible-set geometry",
                "objective_constraint_delta": "linear EH terms are replaced by nonlinear harvested-energy constraints",
                "algorithmic_delta": "SCA is used to handle the nonlinear EH/rate coupling",
                "main_risk": "prior work already covers it",
            },
            "proof_contract": {
                "target_claims": ["monotone SCA convergence"],
                "assumptions": ["compact feasible set"],
                "route": "MM/SCA",
                "algorithmic_route": "solve convex surrogate each iteration",
                "allowed_approximations": ["first-order lower bound"],
            },
            "validation_contract": {
                "metrics": ["U"],
                "figures": ["U versus P_max", "U versus b"],
                "parameter_sweeps": ["P_max", "b"],
                "expected_trends": ["U increases with P_max"],
                "evidence_questions": ["Does U improve in nonlinear EH-sensitive regimes?"],
            },
            "kill_criteria": ["no measurable mechanism-driven gain"],
            "handoff_notes": {
                "phase2_instructions": ["keep SWIPT scope"],
                "interface_warnings": ["do not add RIS"],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            client = FakeClient([research_payload, direction_contract_payload])
            old_minimum = os.environ.get("WARA_PHASE1_REFERENCE_MIN")
            old_search = os.environ.get("WARA_PHASE1_LITERATURE_SEARCH")
            os.environ["WARA_PHASE1_REFERENCE_MIN"] = "0"
            os.environ["WARA_PHASE1_LITERATURE_SEARCH"] = "0"
            try:
                result = run_wara_phase1(
                    topic="SWIPT nonlinear EH",
                    run_dir=tmp_path / "wara-phase1-test",
                    tail_root=tmp_path / "wara_runs" / "phase1_tail",
                    llm_client=client,
                )
            finally:
                if old_minimum is None:
                    os.environ.pop("WARA_PHASE1_REFERENCE_MIN", None)
                else:
                    os.environ["WARA_PHASE1_REFERENCE_MIN"] = old_minimum
                if old_search is None:
                    os.environ.pop("WARA_PHASE1_LITERATURE_SEARCH", None)
                else:
                    os.environ["WARA_PHASE1_LITERATURE_SEARCH"] = old_search

            self.assertEqual(len(client.calls), 2)
            self.assertTrue((result.run_dir / "phase1-4" / "phase1_handoff.json").exists())
            self.assertTrue((result.handoff_dir / "phase1_handoff.json").exists())
            payload = json.loads((result.handoff_dir / "phase1_handoff.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phase1_design"], "wara_native_4_phase_controller")
            self.assertEqual(payload["selected_candidate"]["title"], "Nonlinear SWIPT Utility Maximization")
            self.assertTrue((result.handoff_dir / "topic_focused_literature.json").exists())
            self.assertTrue((result.run_dir / "phase1-1" / "research_object.json").exists())
            self.assertTrue((result.run_dir / "phase1-3" / "contract_bundle.json").exists())
            review = json.loads((result.run_dir / "phase1-3" / "candidate_review.json").read_text(encoding="utf-8"))
            self.assertTrue(review["mechanism_logic_review"][0]["passes_no_pileup_check"])
            manifest_path = result.run_dir / "phase1_controller_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["controller_version"], "wara_phase1_controller_v1")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                {gate["id"] for gate in manifest["gates"]},
                {
                    "scope_gate",
                    "research_object_gate",
                    "evidence_grounding_gate",
                    "direction_contract_gate",
                    "reference_bank_gate",
                    "handoff_gate",
                },
            )
            self.assertIn(
                "phase1_handoff",
                {item["artifact_id"] for item in manifest["frozen_contracts"]},
            )

    def test_validate_handoff_payload_rejects_missing_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_handoff_payload({"selected_candidate": {"title": "weak"}})

    def test_phase1_prompts_do_not_assign_empirical_comparisons(self) -> None:
        scope = {"scope_contract": {"user_topic": "wireless power control"}}
        research_payload = {
            "topic_profile": {},
            "research_object": {
                "research_question": "How should wireless power be controlled?",
                "physical_mechanism": "interference coupling",
                "expected_research_gain": "higher utility",
            },
            "wireless_system_seed": {},
            "mechanism_hypothesis": {},
            "phase2_readiness": {},
            "direction_constraints": {},
        }
        object_system, object_user = build_research_object_prompt("wireless power control", scope)
        frame = build_research_frame_payload("wireless power control", scope, research_payload)
        direction_system, direction_user = build_direction_contract_prompt("wireless power control", frame, {})
        text = "\n".join([object_system, object_user, direction_system, direction_user]).lower()
        self.assertNotIn("baseline", text)
        self.assertNotIn("benchmark", text)
        self.assertIn("hard theorem", text)
        self.assertIn("mechanism extension policy", text)
        self.assertIn("resource coupling change", text)
        self.assertIn("title-only combinations", text)
        self.assertIn("positive mechanism-design goal", text)
        self.assertIn("joint optimization exploit coupling", text)
        self.assertIn("candidate extension axes", text)
        self.assertIn("llm must select", text)
        self.assertIn("changes the marginal value", text)
        self.assertIn("combination novelty", text)
        self.assertIn("why_not_keyword_stacking", text)
        self.assertIn("optimization-centered", text)
        self.assertIn("objective_constraint_structure", text)
        self.assertIn("selection rubric priority", text)
        self.assertIn("concrete research gap first", text)
        self.assertIn("plain and extension candidates must be evaluated by the same rubric", text)
        self.assertIn("plain candidate is a serious candidate", text)
        self.assertIn("quality must be built at the source", text)
        self.assertIn("first-pass research-quality contract", text)
        self.assertIn("first-pass direction-quality contract", text)
        self.assertIn("do not rely on later phases to invent the contribution", text)
        self.assertIn("do not choose a device-response or model-fidelity mechanism by default", text)
        self.assertIn("do not silently default to minimizing bs/transmit power", text)
        self.assertIn("transmit power as a resource cost or diagnostic", text)
        self.assertIn("objective orientation is a deliberate research decision", text)
        self.assertIn("do not select transmit-power minimization simply because it is easy", text)
        self.assertIn("treat model-fidelity choices as modeling choices", text)
        self.assertNotIn("linear-versus-nonlinear", text)
        self.assertNotIn("nonlinear or hardware-response", text)
        self.assertNotIn("prefer a mechanism-enhancing candidate", text)
        self.assertNotIn("do not select the plain candidate by default", text)
        self.assertNotIn("for broad isacp topics, strongly consider whether nonlinear energy harvesting", text)

    def test_broad_isacp_scope_allows_controlled_extension_candidates(self) -> None:
        broad = build_wireless_scope_context("integrated sensing communication and powering")["scope_contract"]
        self.assertTrue(broad["mechanism_extension_policy"]["enabled"])
        self.assertEqual(broad["candidate_extension_mechanisms"], [])
        axis_ids = {axis["id"] for axis in broad["candidate_extension_axes"]}
        axis_order = [axis["id"] for axis in broad["candidate_extension_axes"]]
        self.assertIn("energy_or_hardware_response", axis_ids)
        self.assertIn("propagation_or_spatial_geometry", axis_ids)
        self.assertIn("uncertainty_or_reliability", axis_ids)
        self.assertNotEqual(axis_order[0], "energy_or_hardware_response")
        rules = " ".join(broad["mechanism_extension_policy"]["selection_rules"]).lower()
        self.assertIn("neither plain nor extension candidates win by default", rules)
        self.assertIn("strongest research gap", rules)
        self.assertIn("do not choose a model-fidelity or device-response tweak", rules)
        self.assertNotIn("prefer a mechanism-enhancing candidate", rules)
        self.assertNotIn("RIS", broad["forbidden_added_mechanisms"])

        narrow = build_wireless_scope_context("SWIPT nonlinear EH")["scope_contract"]
        self.assertFalse(narrow["mechanism_extension_policy"]["enabled"])
        self.assertEqual(narrow["candidate_extension_axes"], [])
        self.assertEqual(narrow["candidate_extension_mechanisms"], [])
        self.assertIn("RIS", narrow["forbidden_added_mechanisms"])

        generic_broad = build_wireless_scope_context("queue-aware wireless scheduling optimization")["scope_contract"]
        self.assertTrue(generic_broad["mechanism_extension_policy"]["enabled"])
        self.assertEqual(generic_broad["candidate_extension_mechanisms"], [])
        self.assertGreaterEqual(len(generic_broad["candidate_extension_axes"]), 3)

        broad_near_field = build_wireless_scope_context("near-field communications")["scope_contract"]
        self.assertTrue(broad_near_field["mechanism_extension_policy"]["enabled"])
        self.assertIn("access_or_resource_granularity", {axis["id"] for axis in broad_near_field["candidate_extension_axes"]})
        self.assertIn("energy_or_hardware_response", {axis["id"] for axis in broad_near_field["candidate_extension_axes"]})
        self.assertIn("RIS", broad_near_field["forbidden_added_mechanisms"])

    def test_taxonomy_uses_open_slots_instead_of_default_wireless_template(self) -> None:
        plan = build_wireless_topic_taxonomy_plan("wireless scheduling optimization")
        self.assertEqual(plan["recommended_layers"]["technology"], [])
        self.assertEqual(plan["recommended_layers"]["scenario"], [])
        self.assertEqual(plan["recommended_layers"]["optimization"], ["scheduling"])
        self.assertNotIn("mimo", plan["recommended_layers"]["technology"])
        self.assertNotIn("swipt", plan["recommended_layers"]["technology"])
        self.assertNotIn("sca", plan["recommended_layers"]["theory"])
        self.assertIn("technology", plan["blueprints"][0]["open_slots"])
        self.assertIn("theory", plan["blueprints"][0]["open_slots"])
        self.assertEqual(len(plan["research_axes"]), 10)
        axis_ids = {axis["id"] for axis in plan["research_axes"]}
        self.assertIn("deployment_topology", axis_ids)
        self.assertIn("node_antenna_architecture", axis_ids)
        self.assertIn("propagation_spectrum_regime", axis_ids)
        self.assertIn("algorithm_theory_level", axis_ids)
        self.assertIn("coverage checklist", plan["prompt_block"].lower())
        self.assertIn("not limited to these", plan["prompt_block"].lower())
        self.assertNotIn("hardcoded MIMO/SWIPT/SCA defaults", plan["prompt_block"])

    def test_taxonomy_recognizes_wireless_levels_without_collapsing_them(self) -> None:
        miso = build_wireless_topic_taxonomy_plan("point-to-point MISO SWIPT power control")
        self.assertEqual(miso["recommended_layers"]["technology"], ["miso", "swipt"])
        self.assertEqual(miso["recommended_layers"]["scenario"], ["point_to_point"])
        self.assertEqual(miso["recommended_layers"]["optimization"], ["power_allocation"])

        near_field = build_wireless_topic_taxonomy_plan("near-field XL-MIMO ISAC beamforming")
        self.assertIn("xl_mimo", near_field["recommended_layers"]["technology"])
        self.assertIn("near_field", near_field["recommended_layers"]["scenario"])
        self.assertIn("beamforming_precoding", near_field["recommended_layers"]["optimization"])

        access = build_wireless_topic_taxonomy_plan("UAV-aided NOMA MEC trajectory optimization")
        self.assertIn("noma", access["recommended_layers"]["technology"])
        self.assertIn("uav_aided", access["recommended_layers"]["scenario"])
        self.assertIn("trajectory_design", access["recommended_layers"]["optimization"])

    def test_phase1_google_scholar_source_is_opt_in(self) -> None:
        old_value = os.environ.pop("WARA_PHASE1_GOOGLE_SCHOLAR", None)
        old_override = os.environ.pop("WARA_PHASE1_LITERATURE_SOURCES", None)
        old_ieee = os.environ.pop("WARA_PHASE1_ENABLE_IEEE_XPLORE", None)
        old_openalex = os.environ.pop("WARA_PHASE1_ENABLE_OPENALEX", None)
        old_disable_openalex = os.environ.pop("WARA_PHASE1_DISABLE_OPENALEX", None)
        old_arxiv = os.environ.pop("WARA_PHASE1_ENABLE_ARXIV", None)
        old_disable_arxiv = os.environ.pop("WARA_PHASE1_DISABLE_ARXIV", None)
        old_wcl_disable_arxiv = os.environ.pop("WCL_DISABLE_ARXIV", None)
        old_s2 = os.environ.pop("WARA_PHASE1_ENABLE_SEMANTIC_SCHOLAR", None)
        try:
            sources = resolve_phase1_literature_sources()
            self.assertEqual(sources, ("semantic_scholar", "openalex", "crossref"))
            self.assertNotIn("google_scholar", sources)
            self.assertEqual(build_literature_source_status(sources)["google_scholar"], "available_opt_in")

            os.environ["WARA_PHASE1_DISABLE_OPENALEX"] = "1"
            sources = resolve_phase1_literature_sources()
            self.assertEqual(sources, ("semantic_scholar", "crossref"))
            os.environ.pop("WARA_PHASE1_DISABLE_OPENALEX", None)

            os.environ["WARA_PHASE1_ENABLE_ARXIV"] = "1"
            sources = resolve_phase1_literature_sources()
            self.assertEqual(sources, ("semantic_scholar", "openalex", "crossref", "arxiv"))
            os.environ.pop("WARA_PHASE1_ENABLE_ARXIV", None)

            os.environ["WARA_PHASE1_GOOGLE_SCHOLAR"] = "1"
            sources = resolve_phase1_literature_sources()
            self.assertIn("google_scholar", sources)
            self.assertEqual(build_literature_source_status(sources)["google_scholar"], "enabled_unstable_scraping")
            self.assertEqual(normalize_literature_source("Google Scholar"), "google_scholar")
        finally:
            if old_value is None:
                os.environ.pop("WARA_PHASE1_GOOGLE_SCHOLAR", None)
            else:
                os.environ["WARA_PHASE1_GOOGLE_SCHOLAR"] = old_value
            if old_override is None:
                os.environ.pop("WARA_PHASE1_LITERATURE_SOURCES", None)
            else:
                os.environ["WARA_PHASE1_LITERATURE_SOURCES"] = old_override
            if old_ieee is None:
                os.environ.pop("WARA_PHASE1_ENABLE_IEEE_XPLORE", None)
            else:
                os.environ["WARA_PHASE1_ENABLE_IEEE_XPLORE"] = old_ieee
            if old_openalex is None:
                os.environ.pop("WARA_PHASE1_ENABLE_OPENALEX", None)
            else:
                os.environ["WARA_PHASE1_ENABLE_OPENALEX"] = old_openalex
            if old_disable_openalex is None:
                os.environ.pop("WARA_PHASE1_DISABLE_OPENALEX", None)
            else:
                os.environ["WARA_PHASE1_DISABLE_OPENALEX"] = old_disable_openalex
            if old_arxiv is None:
                os.environ.pop("WARA_PHASE1_ENABLE_ARXIV", None)
            else:
                os.environ["WARA_PHASE1_ENABLE_ARXIV"] = old_arxiv
            if old_disable_arxiv is None:
                os.environ.pop("WARA_PHASE1_DISABLE_ARXIV", None)
            else:
                os.environ["WARA_PHASE1_DISABLE_ARXIV"] = old_disable_arxiv
            if old_wcl_disable_arxiv is None:
                os.environ.pop("WCL_DISABLE_ARXIV", None)
            else:
                os.environ["WCL_DISABLE_ARXIV"] = old_wcl_disable_arxiv
            if old_s2 is None:
                os.environ.pop("WARA_PHASE1_ENABLE_SEMANTIC_SCHOLAR", None)
            else:
                os.environ["WARA_PHASE1_ENABLE_SEMANTIC_SCHOLAR"] = old_s2

    def test_evidence_filter_drops_off_domain_high_citation_hits(self) -> None:
        papers = [
            Paper(
                paper_id="bad",
                title="40 years of cognitive architectures: core cognitive abilities and practical applications",
                year=2018,
                abstract="A survey of symbolic cognitive architectures and practical artificial intelligence.",
                venue="Artificial Intelligence Review",
                citation_count=999,
                source="openalex",
            ),
            Paper(
                paper_id="good",
                title="Optimal Beamforming for Multi-Functional Integrated Sensing, Communication, and Powering Systems",
                year=2025,
                abstract="This wireless paper studies ISACP beamforming, RF powering, sensing, and communication constraints.",
                venue="Electronics Letters",
                citation_count=1,
                source="openalex",
            ),
            Paper(
                paper_id="offscope",
                title="Beyond Diagonal Intelligent Reflecting Surface Aided Integrated Sensing and Communication",
                year=2025,
                abstract="RIS-aided ISAC beamforming with reconfigurable intelligent surfaces.",
                venue="arXiv",
                citation_count=10,
                source="arxiv",
            ),
            Paper(
                paper_id="offdomain",
                title="3D Super-Resolution Ultrasound with Adaptive Weight-Based Beamforming",
                year=2022,
                abstract="Ultrasound imaging with adaptive beamforming.",
                venue="eess.IV",
                citation_count=20,
                source="arxiv",
            ),
            Paper(
                paper_id="eh",
                title="Nonlinear Energy Harvesting for SWIPT Beamforming",
                year=2021,
                abstract="Wireless power transfer and SWIPT beamforming with nonlinear energy harvesting saturation.",
                venue="IEEE",
                citation_count=5,
                source="openalex",
            ),
        ]
        filtered = filter_relevant_literature_papers(
            papers,
            topic="integrated sensing communication and powering",
            research_payload={
                "topic_profile": {
                    "candidate_extension_mechanisms": ["nonlinear energy harvesting"],
                    "forbidden_added_mechanisms": ["RIS"],
                },
                "research_object": {"physical_mechanism": "shared beamforming covariance"},
            },
            selected={"title": "Nonlinear-EH-aware ISACP beamforming"},
            queries=["ISACP integrated sensing communication powering wireless beamforming optimization"],
        )
        self.assertEqual([paper.paper_id for paper in filtered], ["eh", "good"])


if __name__ == "__main__":
    unittest.main()
