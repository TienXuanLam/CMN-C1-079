# Design Specification — CMN-C1-079

**Multi-Language Customer Service Intent Classification & Routing Agent**

| Item | Value |
|---|---|
| Template ID | `CMN-C1-079` |
| Category | Cat 1 — Generic, industry-agnostic (CMN) |
| L1 Base | `AgentBaseGraph` |
| SDK | `agenticstar-agentcore==1.0.0rc1` |
| Risk | 🟢 LOW |
| Architect verdict | design-ready (2026-05-12) |

---

## 1. Agent Overview

Accepts a raw customer service message in any supported language (Japanese, English,
Chinese, Korean, Vietnamese, Thai — configurable). Detects the language, normalises
the text, classifies intent via multilingual LLM, scores urgency, and produces a
structured routing decision envelope with a full classification audit trail.

No translation step. Classification operates on the native-language input using a
multilingual LLM. Intent taxonomy is fully config-driven — the agent code does not
change between industry deployments.

**Trigger:** inbound CS message (string)
**Output:** routing decision envelope (intent, language, confidence, urgency, routing_target, audit hash)

---

## 2. State Schema (`src/schemas/state.py`)

All fields `Optional` primitives (`str / int / float / bool`).
No Pydantic, dataclass, or arbitrary objects (checkpoint safety — ADR-005).

### Input fields
| Field | Type | Description |
|---|---|---|
| `raw_message` | `str` | Raw CS message. Max 10,000 chars. Never logged (S-4). |
| `operator_config` | `str` | JSON string: runtime config overrides (supported_languages, confidence_threshold, routing_rules, routing_allowlist, intent_taxonomy). Credentials MUST NOT be stored here. |

### Processing fields
| Field | Type | Description |
|---|---|---|
| `detected_language` | `str` | ISO 639-1 code: ja/en/zh/ko/vi/th/unknown |
| `normalised_message` | `str` | Unicode-normalised, control-chars stripped. Used for classification. |
| `intent_class` | `str` | Top-1 intent ID from taxonomy (e.g. complaint, account_inquiry). |
| `intent_candidates` | `str` | JSON string: top-3 [{id, label, score}] from LLM. |
| `confidence_score` | `float` | 0.0–1.0. Below threshold → human_review_required=True. |
| `human_review_required` | `bool` | True when confidence_score < confidence_threshold. Non-suppressible. |
| `urgency_level` | `str` | HIGH / MEDIUM / LOW. Set by UrgencyScoreNode inside MainNode. |
| `routing_target` | `str` | Resolved queue ID from routing allowlist. |

### Output fields
| Field | Type | Description |
|---|---|---|
| `final_output` | `str` | Markdown routing decision report (same content as `formatted_output`). |
| `formatted_output` | `str` | Plain Markdown routing decision report — read by SDK `get_output()` as `result["output"]`. Not JSON; callers read it as text. |

### Input validation fields
| Field | Type | Description |
|---|---|---|
| `input_validation_failed` | `str` | Set to `"true"` by `PreProcessNode` when `raw_message` is empty; short-circuits `MainNode` and makes `PostProcessNode` return guidance text instead of a routing envelope. |

### Error fields
| Field | Type | Description |
|---|---|---|
| `error_code` | `str` | Set on error. See §5 for full code list. |
| `error_message` | `str` | Human-readable description. Always set alongside error_code. |

---

## 3. Node Flow

```
[Caller]
    |
    └─ invoke(user_input=raw_message, input_context={...fallback/operator_config...})
    |
    v
┌──────────────────────────────────┐
│  pre_process · PreProcessNode    │
│  S-5/S-2/S-1 input gates         │
│  Detect language → detected_lang │
│  Normalise text → normalised_msg │
└─────────────────┬────────────────┘
                  │
                  v
┌──────────────────────────────────┐
│  main · MainNode (composite)     │
│  ├─ IntentClassifyNode           │
│  │  Multilingual LLM (Azure)     │
│  │  → intent_class, confidence   │
│  │  → intent_candidates          │
│  │  → human_review_required      │
│  ├─ UrgencyScoreNode             │
│  │  Keyword + intent default     │
│  │  → urgency_level              │
│  └─ RoutingDecisionNode          │
│     S-3 allowlist enforcement    │
│     → routing_target             │
└─────────────────┬────────────────┘
                  │
                  v
┌──────────────────────────────────┐
│  post_process · PostProcessNode  │
│  S-3 output gate                 │
│  PII non-echo gate               │
│  Assemble formatted_output       │
└─────────────────┬────────────────┘
                  │
                  v
[Output: result["output"] = plain Markdown routing decision report (string)]
```

| SDK Slot | Node(s) | Responsibility | Output fields |
|---|---|---|---|
| `pre_process` | `PreProcessNode` | S-5/S-2/S-1 gates, language detect, normalise | detected_language, normalised_message, raw_message, operator_config |
| `main` | `MainNode` → IntentClassify + UrgencyScore + RoutingDecision | LLM classification, urgency, routing | intent_class, confidence_score, urgency_level, routing_target |
| `post_process` | `PostProcessNode` | S-3 gate, PII non-echo, envelope assembly | final_output, formatted_output |

---

## 4. Security Gates

### S-1: Input sanitisation (PreProcessNode)
- `raw_message` empty or None → `S1_EMPTY_INPUT`
- `raw_message` length > 10,000 chars → `S1_INPUT_TOO_LONG`
- Control character injection → `S1_INJECTION_DETECTED`
- Binary / base64 blob → `S1_BINARY_INPUT`

### S-2: Credential scan (PreProcessNode)
- Credential patterns in `operator_config` (via `framework.security.detect_credentials()`) → `S2_CREDENTIAL_DETECTED`
- Note: PII in `raw_message` is **not** blocked at S-2 (required for classification context).
  PII must not be echoed in output (S-3 gate) or logged (S-4 rule).

### S-3: Output gate (PostProcessNode + RoutingDecisionNode)
- Routing target must be in `operator_config.routing_allowlist` → reject if not (`S3_ROUTING_NOT_ALLOWLISTED`)
- `human_review_required` flag non-suppressible — always present in output when confidence < threshold
- PII non-echo: `raw_message` content must not appear in `final_output`
- Credential (via `framework.security.detect_credentials()`) / internal URL patterns in output → `S3_BLOCKED`

### S-4: Audit logging
- Logged: `message_ref_hash` (SHA-256 of raw_message), `detected_language`, `intent_class`,
  `confidence_score`, `urgency_level`, `routing_target`, `utc_timestamp`
- **Never logged:** `raw_message` content, `normalised_message`, `intent_candidates` content

### S-5: Trust gate (PreProcessNode)
- `operator_config` is **optional**. Omitted or empty defaults to `{}` — every node's
  own built-in defaults (`_DEFAULT_ROUTING_RULES`, `_DEFAULT_INTENT_TAXONOMY`, etc.)
  apply, and no elevated trust is required, since an empty config customises nothing.
  (Prior to 2026-08-19 this was unconditionally required, which combined with the fix
  below to make the agent unusable by its own declared `VERIFIED_EXTERNAL` entry point
  for the ordinary, non-customising case — fixed the same day.)
- Invalid JSON, or a JSON value that isn't an object (e.g. `"[]"`) → `S5_PERMISSION_DENIED`.
- A **non-empty** `operator_config` requires `caller_trust_level` (state field,
  framework-established from `InvocationContext` — never a self-declared field inside
  `operator_config` itself) to be `INTERNAL` → `S5_PERMISSION_DENIED` otherwise. This is
  the framework's Channel 4 Rule 2 (caller-supplied-claim input channel): a non-empty
  `operator_config` sets `routing_rules`/`routing_allowlist`/`intent_taxonomy`/
  `confidence_threshold`, so a node gated on it must require `INTERNAL`, not
  `VERIFIED_EXTERNAL` — otherwise any authenticated caller could self-declare privileged
  config and there would be no narrowing at all. (Previously this gate read a
  `trust_level` field from inside the caller-controlled `operator_config` JSON itself —
  a caller could grant themselves `"trust_level": "operator"` with zero authentication;
  fixed 2026-08-19.)
  - **Deployment note:** the standalone `src/api/server.py` extends `_resolve_standalone_trust()`
    with a distinct `STG_INTERNAL_RUNNER_TOKEN` bearer specifically so this path is
    reachable outside the platform gateway (the rule's default assumption is that a
    standalone deployment cannot produce `INTERNAL` and does not forward `input_context`
    at all — this template deliberately extends both, per the rule's own guidance for
    templates that need the claim path).

---

## 5. Error Propagation

Contract: when `error_code` is set and not advisory, all downstream nodes return `{}` immediately.

| Code | Node | Fatal? | Description |
|---|---|---|---|
| `S1_EMPTY_INPUT` | PreProcessNode | ✅ | raw_message empty |
| `S1_INPUT_TOO_LONG` | PreProcessNode | ✅ | > 10,000 chars |
| `S1_BINARY_INPUT` | PreProcessNode | ✅ | Binary/base64 blob |
| `S1_INJECTION_DETECTED` | PreProcessNode | ✅ | Control char injection |
| `S2_CREDENTIAL_DETECTED` | PreProcessNode | ✅ | Credential in operator_config |
| `S2_LLM_ERROR` | IntentClassifyNode | ✅ | LLM call failed or not configured |
| `S3_ROUTING_NOT_ALLOWLISTED` | RoutingDecisionNode | ✅ | Target not in allowlist |
| `S3_BLOCKED` | PostProcessNode | ✅ | PII/credential in output |
| `S5_PERMISSION_DENIED` | PreProcessNode | ✅ | Non-empty `operator_config` submitted by a caller whose `caller_trust_level` is not `INTERNAL`, or `operator_config` is invalid JSON / not a JSON object |
| `LOW_CONFIDENCE_ADVISORY` | IntentClassifyNode | ⚠️ non-fatal | confidence < threshold; pipeline continues, human_review_required=True |

---

## 6. Graph Composition (`src/graph/graph.py`)

`AgentBaseGraph` enforces a fixed 5-stage pipeline with exactly three customisable slots
(`pre_process`, `main`, `post_process`). The 5 conceptual nodes are composed inside those 3 slots:

```python
# src/graph/graph.py
class MultiLangIntentRoutingGraph(AgentBaseGraph):
    def register_nodes(self) -> None:
        super().register_nodes()  # injects InitializeNode + FinalizeNode
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode(
            llm_client=self.config.get("llm"),
            timeout_s=float(self.config.get("timeout_s", 20.0)),
            max_retry=int(self.config.get("max_retry", 0)),
        )
        self._nodes["post_process"] = PostProcessNode()
```

`MainNode` composes `IntentClassifyNode → UrgencyScoreNode → RoutingDecisionNode`
sequentially so all three run inside the single `main` slot, calling each sub-node's
`.execute()` directly (not `__call__()`) — only `PreProcessNode`, `MainNode`, and
`PostProcessNode` appear in `node_history`, not the three composed sub-nodes.

Inputs are read with `user_input` taking precedence — Marketplace only ever populates
`user_input`, never `input_context`, on a real invocation. `input_context.raw_message`
is retained purely as a compatibility fallback for the standalone FastAPI adapter,
which mirrors the message into both fields:

```python
agent.invoke(
    user_input=raw_message,
    input_context={"raw_message": raw_message, "operator_config": operator_config},
    ctx=ctx,
)
```

Data flow:
```
START → initialize → pre_process → main → post_process → finalize → END
                                     ↑ (RETRY, up to max_retry)
                                   pre_process
```

---

## 7. Configuration

### `config/agent.yaml` — SDK manifest

```yaml
template_id: CMN-C1-079
name: cmn-c1-079
version: "1.0.0"
description: Multilingual customer service intent classification and routing agent (ja/en/zh/ko/vi/th).
category: 1

requires:
  extras: [openai]
  secrets:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_DEPLOYMENT
```

### `config/config.yaml` — runtime parameters

```yaml
max_retry: 2
memory_enabled: false
timeout_s: 30
```

> Runtime operator overrides (intent taxonomy, routing rules, allowlist, confidence threshold)
> are passed at invocation time via `input_context.operator_config` — not in config files.

---

## 8. Output Envelope (`result["output"]`)

`invoke()` returns `result["output"]` which is `formatted_output` — a plain
**Markdown string** (`PostProcessNode` assembles it via `"\n".join(sections)`),
not a dict and not JSON. Callers read it as text, e.g.:

```markdown
# Customer Service Routing Decision

## Classification

- **Detected language:** `ja`
- **Intent:** `complaint`
- **Confidence:** 92.0%
- **Urgency:** **HIGH**
- **Human review:** Not required

## Routing

Route this request to **`queue_complaints`**.

## Candidate Intents

- `complaint` — 苦情 (92.0%)
- `escalation` — エスカレーション (6.0%)
- `general_inquiry` — 一般照会 (2.0%)

## Audit

- **Message reference:** `<sha256>`
- **Timestamp:** `2026-05-27T03:00:00Z`

> The original customer message is intentionally excluded from this report.
```

On error: `result["status"] == "error"`, `result["output"]` is `None`.

---

## 9. Cat 1 → Cat 2 Upgrade Path

This template is Cat 1 (CMN). Industry Cat 2 variants customise only:
- `intent_taxonomy` (add industry-required intents e.g. FIN: `suitability_check`)
- `routing_rules` and `routing_allowlist`
- `urgency_keywords` per language (optional)
- `supported_languages` (e.g. VI for FPT Vietnam deployment)

No code changes needed — configuration only.

---

## 10. Definition of Done

- [x] `pyproject.toml` — SDK dependency removed; installed separately via deploy token
- [x] `config/agent.yaml` — SDK manifest format (template_id, name, version, category, requires)
- [x] `config/config.yaml` — runtime parameters (max_retry, timeout_s, memory_enabled)
- [x] `src/schemas/state.py` — flat TypedDict extending AgentState; all fields Optional primitives
- [x] 5 nodes migrated to `FunctionNode`; `execute(state) -> dict`; `emit_trace_event` from `shared.utils.audit_logger`
- [x] `src/graph/graph.py` — SDK `AgentBaseGraph` with `_nodes` slots (pre_process / main / post_process)
- [x] `src/api/server.py` — `bound_secrets` + `provision_secrets` pattern; `POST /invoke`, `GET /health`
- [x] Legacy pre-wheel framework provisioning removed
- [x] Tests migrated to `InMemoryProvider` + `bind_secrets`; `agent.invoke()` pattern
- [x] Scaffold structure complete
- [x] CI pipeline green

---
*Design Specification: CMN-C1-079 · eng-moderator · 2026-05-27 · updated post-migration*
