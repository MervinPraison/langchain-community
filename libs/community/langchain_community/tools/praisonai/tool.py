"""PraisonAI tools for running multi-agent workflows."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field


class PraisonAITool(BaseTool):
    """Tool for running queries through a PraisonAI multi-agent workflow.

    Example usage:
    .. code-block:: python

        tool = PraisonAITool(api_url="http://localhost:8080")
        result = tool.run("Research AI trends")
    """

    name: str = "praisonai"
    description: str = (
        "A multi-agent AI workflow tool. "
        "Useful for complex tasks that benefit from multiple AI agents "
        "working together. Input should be a query or task description."
    )
    api_url: str = Field(default="http://localhost:8080")
    timeout: int = Field(default=300)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the query through PraisonAI."""
        response = httpx.post(
            f"{self.api_url}/agents",
            json={"query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")


class PraisonAIAgentTool(BaseTool):
    """Tool for running queries through a specific PraisonAI agent.

    Example usage:
    .. code-block:: python

        tool = PraisonAIAgentTool(
            api_url="http://localhost:8080",
            agent_name="researcher"
        )
        result = tool.run("Find latest AI papers")
    """

    name: str = "praisonai_agent"
    description: str = (
        "Run a query through a specific AI agent. "
        "Input should be a query or task description."
    )
    api_url: str = Field(default="http://localhost:8080")
    agent_name: str = Field(description="Name of the agent to use")
    timeout: int = Field(default=300)

    def __init__(self, agent_name: str, **kwargs: Any) -> None:
        """Initialize with agent name."""
        super().__init__(agent_name=agent_name, **kwargs)
        self.name = f"praisonai_{agent_name}"
        self.description = (
            f"Run a query through the '{agent_name}' AI agent. "
            "Input should be a query or task description."
        )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the query through the specific agent."""
        response = httpx.post(
            f"{self.api_url}/agents/{self.agent_name}",
            json={"query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")
