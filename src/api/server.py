"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly.

import asyncio
import logging
import os
import secrets
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.secrets import factory as secrets_factory
from shared.secrets.inmemory_provider import InMemoryProvider
from src.graph.graph import MultiLangIntentRoutingGraph

logger = logging.getLogger(__name__)

app = FastAPI(title="MultiLangIntentRoutingAgent")

# Same config_dir / "config.yaml" convention as AgentRegistry._compile_and_cache()
# (mediator/registry/agent_registry.py) — absent config.yaml is tolerated, matching
# the registry's own `if exists() else {}` guard. Without this, the standalone
# adapter always ran with config={}, so hitl.enabled / memory_enabled / max_retry
# etc. silently never reached Graph() on this path.
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}

_configured_secrets_provider = secrets_factory(namespace="cmn", agent_name="cmn-c1-079")
_azure_secret_keys = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)
_secrets_provider = InMemoryProvider(
    {
        key: value
        for key in _azure_secret_keys
        if (value := os.environ.get(key) or _configured_secrets_provider.get(key)) is not None
    },
    namespace="cmn",
    agent_name="cmn-c1-079",
)

agent = MultiLangIntentRoutingGraph(config=_config)

# Mirror AgentRegistry._compile_and_cache()'s conditional checkpointer — hitl.enabled
# or memory_enabled needs one, or interrupt()/memory silently no-ops on this path.
# NOT CheckpointerFactory (mediator/factory/checkpointer_factory.py): mediator* is
# excluded from the published wheel, so templates cannot import it. This process
# runs exactly one agent, so a private MemorySaver per process is the standalone
# equivalent of the registry's cross-agent singleton-eviction concern.
_hitl_enabled = agent.config.get("hitl", {}).get("enabled", False)
_needs_checkpointer = agent.config.get("memory_enabled") or _hitl_enabled
agent.compile(checkpointer=MemorySaver() if _needs_checkpointer else None)

try:
    agent.provision_secrets(_secrets_provider)
except Exception as exc:
    logger.error(
        "cmn-c1-079: failed to provision secrets at startup — "
        "requests will fail until secrets are available. Error: %s",
        exc,
    )


class InvokeRequest(BaseModel):
    """Public contract: one customer service message and optional operator config."""

    model_config = ConfigDict(extra="forbid")

    input: str
    session_id: str = ""
    operator_config: str = ""


def _bearer_matches(supplied: str, expected: str) -> bool:
    """Constant-time bearer comparison that is safe for non-ASCII header input."""
    return secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel, authorization: str, invoke_auth_token: str | None, internal_runner_token: str | None
) -> TrustLevel:
    """Authenticate standalone callers without allowing external-token elevation.

    STG_INTERNAL_RUNNER_TOKEN is a distinct, CI-generated deployment credential.
    It is considered only for an anonymous caller and maps exactly to INTERNAL;
    INVOKE_AUTH_TOKEN remains VERIFIED_EXTERNAL. Middleware-established trust is
    never changed. When either token is configured on the deployment and the
    supplied bearer matches neither, the request is rejected (401) rather than
    silently falling back to ANONYMOUS — a caller must be provably unauthenticated
    (no tokens configured at all) to receive the ANONYMOUS floor.
    """
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    # This adapter is the entry-point auth boundary (standalone equivalent of
    # platform AuthMiddleware). Both tokens are deployment-level caller credentials,
    # not agent secrets: no InvocationContext exists before this boundary, so
    # ctx.secrets cannot apply.
    trust = _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        # agent.invoke() is synchronous and blocking (LangGraph's
        # _compiled.invoke(), which waits on IntentClassifyNode's LLM call
        # for up to _LLM_TIMEOUT_SECONDS). Calling it directly inside this
        # `async def` handler with no `await` freezes the whole uvicorn
        # event loop for that entire duration -- every other in-flight
        # request (including /health) stalls until this one finishes.
        # asyncio.to_thread (not loop.run_in_executor) is required here: it
        # copies the current contextvars.Context into the worker thread, so
        # the bound_secrets() ContextVar set just above is still visible to
        # IntentClassifyNode._build_llm()'s ctx.secrets.require() calls --
        # framework/secrets/context.py's own docstring documents this exact
        # propagation boundary. run_in_executor does not copy the context
        # and would make every secret lookup fail with MissingSecret.
        result = await asyncio.to_thread(
            agent.invoke,
            user_input=req.input,
            input_context={
                "raw_message": req.input,
                "operator_config": req.operator_config,
            },
            ctx=ctx,
        )
        return cast("dict[str, Any]", result)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "cmn-c1-079"}
