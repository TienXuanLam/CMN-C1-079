# Test Specification — CMN-C1-079

**Multi-Language Customer Service Intent Classification & Routing Agent**

---

## 1. Overview

| Item | Value |
|---|---|
| Template ID | CMN-C1-079 |
| Stage | Implementation → Test |
| Unit test file | `tests/unit/test_cmn_c1_079.py` |
| Unit test file (MainNode) | `tests/unit/test_main_node.py` |
| PB test files | `tests/proof_of_boundary/test_pb_cmn_c1_079.py`, `test_import_isolation.py`, `test_state_safety.py` |
| Integration test | `tests/integration/test_graph_smoke.py` |

All tests use `InMemoryProvider` + `bind_secrets` (no legacy provisioning shims, no real LLM calls).
LLM-dependent test cases use a fixed-response mock implementing `BaseLLM.complete()`.

---

## 2. Unit Test Cases (`test_cmn_c1_079.py`)

| TC | Description | Input | Expected |
|---|---|---|---|
| TC-01 | State fields are Optional primitives | `MultiLangIntentRoutingState` type hints | All agent-specific fields are `Optional[str\|int\|float\|bool]` |
| TC-02 | S-1: empty raw_message → safe guidance, not a fatal error | `raw_message=""` | `result["status"] == "success"`, output contains "Customer service message required" |
| TC-03 | S-1: oversized input | `raw_message="x"*10001` | `result["status"] == "error"` |
| TC-04 | S-1: binary blob | base64-like string | `result["status"] == "error"` |
| TC-05 | S-1: control char injection | `"hello\x00world"` | `result["status"] == "error"` |
| TC-06 | S-2: credential in operator_config | `api_key=sk-...` in config | `result["status"] == "error"` |
| TC-07 | S-5: missing operator_config | empty operator_config | `result["status"] == "error"` |
| TC-08 | S-5: invalid trust_level | `trust_level: "user"` | `result["status"] == "error"` |
| TC-09 | Error propagation: fatal code stops downstream | control-char injection (`S1_INJECTION_DETECTED`, fatal) | `result["status"] == "error"`, `result["output"]` absent |
| TC-10 | Happy path: Japanese complaint | ja complaint text + HIGH keyword | `result["output"]["routing_target"] == "queue_escalation"`, `detected_language == "ja"` |
| TC-11 | Happy path: English account inquiry | en account text | `detected_language == "en"`, `routing_target` in selfservice/ops |
| TC-12 | Happy path: Vietnamese general inquiry | vi text with diacritics | `detected_language == "vi"` |
| TC-13 | Low confidence → human_review | mock confidence=0.50 | `result["output"]["human_review_required"] == True` |
| TC-14 | Routing target not in allowlist | narrow allowlist, general intent | `result["status"] == "error"` |
| TC-15 | PII not echoed in output | message with PII text | raw_message absent from `result["output"]` |
| TC-16 | LLM auth/permission error → fatal `S2_LLM_ERROR` (internal to IntentClassifyNode) | mock LLM raises an auth-flavoured RuntimeError | `result["status"] == "error"` |
| TC-16b | LLM timeout → non-fatal guidance, not indistinguishable from input error | mock LLM raises a timeout-flavoured RuntimeError | `result["status"] == "success"`, output contains "Classification timed out" |
| TC-17 | Import isolation: no agenticstar in src/ | AST scan of `src/` | 0 violations |
| TC-18 | execute() contract: no _invoke_impl | grep `src/` | 0 occurrences |
| TC-19 | Graph compile + invoke smoke | valid input | `result["status"] == "success"`, `result["output"]` not None |
| TC-20 | Unknown language | Cyrillic text | `result["output"]["detected_language"] == "unknown"` |
| TC-S1 | `_run_input_gates()` rejects empty input | `raw_message=""` | returns dict with `error_code == S1_EMPTY_INPUT` |
| TC-S2 | `_run_input_gates()` rejects missing config | empty operator_config | returns dict with `error_code == S5_PERMISSION_DENIED` |
| TC-S3 | `_run_output_gate()` blocks PII echo | output containing raw_message | returns dict with `error_code == S3_BLOCKED` |
| TC-S4 | `required_trust_level` declared on PreProcessNode | class attribute check | `TrustLevel.VERIFIED_EXTERNAL` |
| TC-R3-7 | BaseLLM imported in intent_classify | AST scan | `from shared.services.llm.base_llm import BaseLLM` found |

---

## 3. MainNode Unit Tests (`test_main_node.py`)

| TC | Description | Expected |
|---|---|---|
| success: account inquiry | mock LLM returns account_inquiry | `intent_class == "account_inquiry"`, routing in selfservice/ops |
| success: complaint routes escalation | mock + urgency keyword | `routing_target == "queue_escalation"` |
| low confidence → human_review | mock confidence=0.50 | `human_review_required == True`, `error_code == LOW_CONFIDENCE_ADVISORY` |
| no LLM client → S2_LLM_ERROR | `llm_client=None` | `error_code == S2_LLM_ERROR` |
| LLM exception → S2_LLM_ERROR | mock raises RuntimeError | `error_code == S2_LLM_ERROR` |
| fatal upstream error propagated | `error_code=S1_EMPTY_INPUT` in state | `result == {}` |
| non-allowlisted routing → error | narrow allowlist | `error_code == S3_ROUTING_NOT_ALLOWLISTED` |
| execute() contract | inspect signature | params[1] == "state", no `_invoke_impl` |

---

## 4. Proof-of-Boundary Tests

| PB | File | Boundary | Expected |
|---|---|---|---|
| PB-1 | `test_pb_cmn_c1_079.py` | Output is JSON-serializable | `result["output"]` serializable as dict |
| PB-2 | `test_pb_cmn_c1_079.py` | S-3 PII non-echo | raw_message absent from `result["output"]` |
| PB-3 | `test_pb_cmn_c1_079.py` | S-3 routing allowlist | Non-allowlisted target → `result["status"] == "error"` |
| PB-4 | `test_import_isolation.py` | L0 import isolation | 0 agenticstar imports in src/ or tests/ |
| PB-5 | `test_pb_cmn_c1_079.py` | S-1 injection gate | Injection blocked → `result["status"] == "error"` |
| PB-6 | `test_pb_cmn_c1_079.py` | Gate execution order | `_run_input_gates` called before `_run_output_gate` |
| PB-7 | `test_import_isolation.py` | L1 framework imports present | At least one `framework.*` import in src/ |
| PB-8 | `test_import_isolation.py` | L2 shared imports present | At least one `shared.*` import in src/ |
| PB-9 | `test_state_safety.py` | No credential field names | No `api_key`, `secret`, `password` etc. in state fields |
| PB-10 | `test_state_safety.py` | Msgpack-safe types only | No Pydantic/BaseModel in state annotations |
| PB-11 | `test_state_safety.py` | All domain fields declared | 13 required fields present in `MultiLangIntentRoutingState` |
| PB-12 | `test_state_safety.py` | State is TypedDict, not Pydantic | `isinstance` check at runtime |

---

## 5. Integration Tests (`test_graph_smoke.py`)

| TC | Description | Expected |
|---|---|---|
| compile succeeds | `agent.compile()` | `agent._compiled is not None` |
| full pipeline English | en account inquiry | `status == "success"`, `detected_language == "en"` |
| full pipeline Japanese | ja complaint | `status == "success"`, `detected_language == "ja"` |
| guidance on empty input (not a fatal error) | `raw_message=""` | `status == "success"`, output contains "Customer service message required" |
| error on missing operator_config | empty config | `status == "error"` |
| node_history contains all nodes | valid input | InitializeNode + PreProcessNode + MainNode + PostProcessNode + FinalizeNode |

---

## 6. Coverage Summary

| Criterion | Tests |
|---|---|
| Security gates S-1, S-2, S-3, S-5 | TC-02–TC-08, TC-S1–TC-S3, PB-2, PB-3, PB-5 |
| Error propagation contract | TC-09, test_main_node fatal propagation |
| Multilingual happy paths (ja/en/vi) | TC-10–TC-12, integration smoke |
| LOW_CONFIDENCE_ADVISORY non-fatal path | TC-13, test_main_node low confidence |
| Import isolation (L0/L1/L2 boundary) | TC-17, PB-4, PB-7, PB-8 |
| execute() contract | TC-18, test_main_node contract |
| State primitive + msgpack safety | TC-01, PB-9–PB-12 |
| Full pipeline end-to-end | integration smoke (6 TCs) |

---
*Test Specification: CMN-C1-079 · updated post-migration · 2026-06-22*
