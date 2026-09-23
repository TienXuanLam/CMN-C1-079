# CMN-C1-079's server.py must build its secrets provider before Graph() and
# never cache a real, secret-bound LLM client on a shared node instance
# (per the framework's LLM-injection guidance) — a node instance is shared across every invocation
# served from the registry's LRU cache, so a client built once at server
# boot from ctx.secrets would be reused by every subsequent request,
# defeating secret rotation and risking cross-invocation credential reuse in
# a multi-tenant runtime. server.py therefore never sets config["llm"] in
# production; IntentClassifyNode._build_llm() resolves the Azure OpenAI
# secrets (AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_DEPLOYMENT)
# fresh, per invocation, via ctx.secrets.require() inside execute(). Missing
# Azure secrets at boot must never crash import — deploy-stg provisions no
# key today.

import importlib

_AZURE_SECRET_KEYS = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT")


class TestServerBootsWithoutOpenAIKey:
    def test_server_imports_and_app_constructs_with_no_key(self, monkeypatch):
        for key in _AZURE_SECRET_KEYS:
            monkeypatch.delenv(key, raising=False)

        import src.api.server as server

        importlib.reload(server)

        assert server.app is not None
        assert server.agent is not None


class TestServerNeverCachesRealLlmClient:
    def test_server_has_no_llm_attribute(self, monkeypatch):
        """server.py must not build/hold a real LLM client at module scope.

        Absence of a module-level `_llm` attribute is the regression guard:
        a prior implementation built an OpenAIClient once at boot and passed
        it into every node via config["llm"], which is exactly the
        shared-cached-credential hazard this fix removes.
        """
        from shared.secrets.inmemory_provider import InMemoryProvider

        monkeypatch.setattr(
            "shared.secrets.factory",
            lambda namespace, agent_name: InMemoryProvider({key: "dummy-test-value" for key in _AZURE_SECRET_KEYS}),
        )

        import src.api.server as server

        importlib.reload(server)

        assert not hasattr(server, "_llm")
        assert server.agent._nodes["main"]._intent_classify._llm is None

    def test_intent_classify_resolves_llm_lazily_from_ctx_secrets(self):
        """IntentClassifyNode._build_llm() resolves the Azure OpenAI secrets
        per call, not once at construction — the actual lazy-resolution path
        server.py relies on instead of constructor injection, and the same
        path that makes the canonical AgentRegistry deployment (which never
        runs server.py at all) work identically."""
        from framework.secrets.context import bound_secrets
        from shared.secrets.inmemory_provider import InMemoryProvider
        from shared.services.llm.azure_openai_client import AzureOpenAIClient

        from src.nodes.intent_classify import IntentClassifyNode

        node = IntentClassifyNode()
        assert node._llm is None
        state = {
            "user_input": "",
            "input_context": {},
            "caller_trust_level": "INTERNAL",
            "caller_id": "test",
            "correlation_id": "test",
            "session_id": "test",
            "thread_id": "test",
            "trace_id": "",
            "hitl_allowed": True,
            "node_history": [],
            "error_log": [],
            "status": "pending",
            "execution_time": {},
        }
        secrets = InMemoryProvider({key: "dummy-test-value" for key in _AZURE_SECRET_KEYS})
        with bound_secrets(secrets):
            llm = node._build_llm(state)
        assert isinstance(llm, AzureOpenAIClient)
        assert node._llm is None


class TestStandaloneTrustPromotion:
    def test_external_bearer_never_promotes_to_internal(self):
        import src.api.server as server
        from framework.schemas.trust_level import TrustLevel

        assert (
            server._resolve_standalone_trust(TrustLevel.ANONYMOUS, "Bearer external", "external", "runner")
            is TrustLevel.VERIFIED_EXTERNAL
        )

    def test_runner_bearer_promotes_to_internal(self):
        import src.api.server as server
        from framework.schemas.trust_level import TrustLevel

        assert (
            server._resolve_standalone_trust(TrustLevel.ANONYMOUS, "Bearer runner", "external", "runner")
            is TrustLevel.INTERNAL
        )

    def test_wrong_or_missing_bearer_is_rejected_when_auth_is_enabled(self):
        import pytest
        import src.api.server as server
        from fastapi import HTTPException
        from framework.schemas.trust_level import TrustLevel

        for authorization in ("", "Bearer wrong"):
            with pytest.raises(HTTPException) as exc:
                server._resolve_standalone_trust(TrustLevel.ANONYMOUS, authorization, "external", "runner")
            assert exc.value.status_code == 401
