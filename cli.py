# AgentCore Platform v1.0
"""AGENTIC STAR Marketplace entrypoint for CMN-C1-079."""

from pathlib import Path
from typing import Any

from framework.utils.config_loader import load_agent_config
from shared.bootstrap.marketplace_app import run_agent_marketplace
from src.graph.graph import MultiLangIntentRoutingGraph

extend_config: dict[str, Any] = {}

if __name__ == "__main__":
    run_agent_marketplace(
        MultiLangIntentRoutingGraph,
        agent_name="CMN-C1-079",
        namespace="cmn",
        config={**load_agent_config(Path(__file__).resolve().parent), **extend_config},
    )
