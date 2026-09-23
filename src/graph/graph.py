"""AgentCore Platform v1.0"""

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.trust_level import TrustLevel

from src.nodes.pre_process_node import PreProcessNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import MultiLangIntentRoutingState


class MultiLangIntentRoutingGraph(AgentBaseGraph):
    """Fixed-pipeline graph for multilingual CS intent classification and routing.

    Slot mapping (5 conceptual nodes → 3 SDK slots):
      pre_process  → PreProcessNode        (S-1/S-2/S-5 gates, lang detect, normalise)
      main         → MainNode              (IntentClassify → UrgencyScore → RoutingDecision)
      post_process → PostProcessNode   (S-3 gate, PII non-echo, envelope assembly)

    Runtime config keys (read from config/config.yaml via self.config):
      timeout_s, max_retry — passed to MainNode at register_nodes() time.
      self.config is populated by cli.py via
      framework.utils.config_loader.load_agent_config(), which reads
      config/config.yaml before run_agent_marketplace() constructs this
      graph.
    """

    # Must match config/agent.yaml's required_trust_level (VERIFIED_EXTERNAL) —
    # AgentRegistry reads the manifest for compile-time gating, but the graph
    # class itself is a separate S-1 declaration point; leaving it undeclared
    # silently inherited BaseGraph's own default rather than enforcing the
    # manifest's stated policy at this layer.
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    @property
    def name(self) -> str:
        return "cmn-c1-079"

    @property
    def state_schema(self) -> type:
        return MultiLangIntentRoutingState

    def register_nodes(self) -> None:
        super().register_nodes()  # injects InitializeNode + FinalizeNode

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode(
            llm_client=self.config.get("llm"),
            timeout_s=float(self.config.get("timeout_s", 20.0)),
            max_retry=int(self.config.get("max_retry", 0)),
        )
        self._nodes["post_process"] = PostProcessNode()
