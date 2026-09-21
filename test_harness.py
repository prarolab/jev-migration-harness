import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from harness import MigrationHarness
from main import run
from models import AssessmentInput, ChoiceAnswer, LLMAnswer
from providers import DemoProviders, LiveProviders, ProviderError


ROOT = Path(__file__).resolve().parent


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
            "LANGCHAIN_TRACING": "false",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.raw = json.loads((ROOT / "sample_input.json").read_text(encoding="utf-8"))
        self.application = AssessmentInput.model_validate(self.raw)
        self.providers = DemoProviders()

    def harness(self, **kwargs):
        return MigrationHarness(self.providers.jev, self.providers.llm, **kwargs)

    def test_demo_routes_and_no_approval(self):
        result = self.harness().chain.invoke(self.application)
        rows = {item["question_id"]: item for item in result["results"]}
        self.assertEqual(rows["ci"]["answer"], "Yes")
        self.assertEqual(rows["ci"]["route"], "jev")
        self.assertIsNone(rows["encryption"]["answer"])
        self.assertEqual(rows["encryption"]["status"], "insufficient")
        self.assertEqual(rows["tomcat"]["route"], "llm")
        self.assertEqual(rows["funding"]["status"], "owner_confirmation_required")
        self.assertEqual(rows["recovery"]["status"], "conflicting")
        self.assertTrue(all(item["review_required"] for item in rows.values()))
        self.assertFalse(result["migration_approved"])
        self.assertEqual(result["metrics"]["jev_calls"], 1)
        self.assertEqual(result["metrics"]["llm_calls"], 2)
        self.assertIsNone(result["metrics"]["actual_cost_usd"])

    def test_llm_baseline_has_same_owner_gate(self):
        result = self.harness(strategy="llm-only").chain.invoke(self.application)
        self.assertEqual(result["metrics"]["jev_calls"], 0)
        self.assertEqual(result["metrics"]["llm_calls"], 4)
        self.assertEqual(result["results"][3]["route"], "human")

    def test_fanout_and_policy_scope(self):
        state = self.harness().prepare(self.application)
        self.assertNotIn("funding", state.request["state"]["items"])
        self.assertNotIn("funding__route", state.request["questions"])
        self.assertIn("ci__route", state.request["questions"])
        self.assertIn("ci__answer", state.request["questions"])
        self.assertNotIn("tomcat__answer", state.request["questions"])
        self.assertEqual(
            set(state.request["questions"]["tomcat__route"]["criteria"]), {"llm", "human"},
        )
        self.assertEqual([item.id for item in state.evidence["ci"]], ["E1"])

    def test_stale_future_and_wrong_environment_excluded(self):
        for field, value in [
            ("observed_at", "2020-01-01"),
            ("observed_at", "2027-01-01"),
            ("environment", "development"),
        ]:
            with self.subTest(field=field, value=value):
                raw = copy.deepcopy(self.raw)
                raw["evidence"][0][field] = value
                result = self.harness().chain.invoke(AssessmentInput.model_validate(raw))
                self.assertEqual(result["results"][0]["status"], "insufficient")
                self.assertIsNone(result["results"][0]["answer"])

    def test_no_evidence_never_means_no(self):
        self.raw["questions"][0]["evidence_ids"] = []
        result = self.harness().chain.invoke(AssessmentInput.model_validate(self.raw))
        self.assertIsNone(result["results"][0]["answer"])
        self.assertEqual(result["results"][0]["route"], "human")

    def test_non_applicable_needs_reason_and_skips_models(self):
        self.raw["questions"][0]["applicable"] = False
        with self.assertRaises(ValidationError):
            AssessmentInput.model_validate(self.raw)
        self.raw["questions"][0]["applicability_reason"] = "Owner-declared scope exclusion."
        result = self.harness().chain.invoke(AssessmentInput.model_validate(self.raw))
        self.assertEqual(result["results"][0]["status"], "not_applicable")

    def test_all_owner_questions_make_no_model_calls(self):
        for question in self.raw["questions"]:
            question["requires_owner"] = True
        result = self.harness().chain.invoke(AssessmentInput.model_validate(self.raw))
        self.assertEqual(self.providers.calls, [])
        self.assertEqual(result["metrics"]["jev_calls"], 0)

    def test_low_confidence_escalates(self):
        result = self.harness(threshold=0.99).chain.invoke(self.application)
        self.assertEqual(result["results"][0]["route"], "llm")
        self.assertEqual(result["results"][4]["status"], "conflicting")

    def test_unknown_candidate_cannot_be_accepted_as_no(self):
        original = self.providers._jev

        def changed(request):
            response = original(request)
            item = response["answers"]["ci__answer"]
            item["choice"] = "__unknown__"
            item["probabilities"] = {"Yes": 0.01, "No": 0.01, "__unknown__": 0.97, "__conflict__": 0.01}
            return response

        self.providers.jev = RunnableLambda(changed)
        result = self.harness().chain.invoke(self.application)
        self.assertEqual(result["results"][0]["route"], "llm")

    def test_llm_budget_leaves_unresolved(self):
        result = self.harness(max_llm_calls=0).chain.invoke(self.application)
        self.assertEqual(result["metrics"]["llm_calls"], 0)
        self.assertEqual(result["results"][1]["status"], "budget_exhausted")
        self.assertIsNone(result["results"][1]["answer"])

    def test_invalid_distribution_rejected(self):
        for probabilities in [{"yes": 1.2, "no": -0.2}, {"yes": 0.1, "no": 0.2}]:
            with self.assertRaises(ValidationError):
                ChoiceAnswer.model_validate({
                    "type": "choice", "choice": "yes",
                    "confidence": 0.9, "probabilities": probabilities,
                })

    def test_missing_jev_answer_fails_explicitly(self):
        original = self.providers._jev

        def incomplete(request):
            response = original(request)
            del response["answers"]["ci__answer"]
            return response

        self.providers.jev = RunnableLambda(incomplete)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.harness().chain.invoke(self.application)

    def test_provider_failure_does_not_become_demo_or_llm_fallback(self):
        def fail(_):
            raise ProviderError("Provider unavailable")

        self.providers.jev = RunnableLambda(fail)
        with self.assertRaises(ProviderError):
            self.harness().chain.invoke(self.application)
        self.assertEqual(self.providers.calls, [])

    def test_bad_llm_citations_and_choices_rejected(self):
        original = self.providers._llm
        for mutation in ["foreign", "fabricated", "wrong_question", "invalid_choice"]:
            def changed(payload, mutation=mutation):
                response = original(payload)
                if mutation == "foreign":
                    response["citations"][0]["evidence_id"] = "E999"
                elif mutation == "fabricated":
                    response["citations"][0]["quote"] = "A sentence not present in any source."
                elif mutation == "wrong_question":
                    response["question_id"] = "other"
                else:
                    response["answer"] = "Maybe"
                return response

            with self.subTest(mutation=mutation):
                self.providers.llm = RunnableLambda(changed)
                with self.assertRaises(ValueError):
                    self.harness(strategy="llm-only").chain.invoke(self.application)

    def test_supported_answer_requires_citations(self):
        with self.assertRaises(ValidationError):
            LLMAnswer.model_validate({
                "question_id": "ci", "status": "supported", "answer": "Yes",
                "citations": [], "missing_evidence": [],
            })

    def test_duplicate_ids_and_bad_refs_rejected(self):
        for mutation in ["question", "evidence", "reference"]:
            raw = copy.deepcopy(self.raw)
            if mutation == "question":
                raw["questions"].append(raw["questions"][0])
            elif mutation == "evidence":
                raw["evidence"].append(raw["evidence"][0])
            else:
                raw["questions"][0]["evidence_ids"] = ["unknown"]
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                AssessmentInput.model_validate(raw)

    def test_oversize_request_fails_before_network(self):
        with self.assertRaisesRegex(ValueError, "never silently truncated"):
            self.harness(max_request_bytes=10).chain.invoke(self.application)
        self.assertEqual(self.providers.calls, [])

    def test_live_requires_explicit_egress_permission(self):
        from argparse import Namespace
        args = Namespace(
            input=ROOT / "sample_input.json", output=None, mode="live",
            allow_external=False, strategy="hybrid", threshold=0.85, max_llm_calls=5,
        )
        with self.assertRaisesRegex(ValueError, "allow-external"):
            run(args)

    def test_http_adapters_end_to_end_without_network(self):
        requests = []
        fixtures = DemoProviders()

        def respond(request):
            requests.append(request)
            body = json.loads(request.content)
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            if request.url.host == "api.typesafe.ai":
                return httpx.Response(200, json=fixtures._jev(body))
            self.assertEqual(request.url.host, "api.openai.com")
            self.assertFalse(body["store"])
            schema = body["response_format"]["json_schema"]["schema"]
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            payload = json.loads(body["messages"][1]["content"])
            return httpx.Response(200, json={
                "model": "mock-llm",
                "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(fixtures._llm(payload)),
                }}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            providers = LiveProviders(client, "test-key", "test-key", "mock-llm")
            result = MigrationHarness(providers.jev, providers.llm).chain.invoke(self.application)
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(providers.calls), 3)
        self.assertEqual(result["results"][2]["route"], "llm")

    def test_http_errors_and_refusals_are_not_accepted(self):
        cases = [
            (429, {"error": "PRIVATE BODY"}, True),
            (302, {}, True),
            (200, {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}, False),
            (200, {"choices": [{"finish_reason": "stop", "message": {"refusal": "No"}}]}, False),
        ]
        for status, response, is_jev in cases:
            with self.subTest(status=status, response=response):
                with httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(status, json=response)
                )) as client:
                    providers = LiveProviders(client, "test-key", "test-key", "mock-llm")
                    with self.assertRaises(ProviderError) as context:
                        (providers.jev if is_jev else providers.llm).invoke({})
                    self.assertNotIn("PRIVATE BODY", str(context.exception))


if __name__ == "__main__":
    unittest.main()
