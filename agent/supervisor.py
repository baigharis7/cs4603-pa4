"""Supervisor node + routing edge (Task 1.3)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

_CALCULATION_KEYWORDS = {
    "calculate",
    "compute",
    "increase",
    "decrease",
    "percentage",
    "percent",
    "growth",
    "cagr",
    "convert",
    "compare",
    "difference",
    "project",
    "multiply",
    "divide",
}


def _fallback_route(step: str) -> str:
    """Use deterministic keywords if the LLM returns an invalid label."""

    normalized = step.lower()
    if any(keyword in normalized for keyword in _CALCULATION_KEYWORDS):
        return MCP
    return RAG


def make_supervisor(llm):
    """Create a supervisor that routes the current plan step."""

    def supervisor(state: AnalystState) -> dict:
        index = state["current_step_index"]
        plan = state["plan"]

        if index >= len(plan):
            return {"next_agent": SYNTH}

        current_step = plan[index]

        response = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=current_step),
            ]
        )

        decision = str(response.content).strip().lower()

        if RAG in decision:
            next_agent = RAG
        elif MCP in decision:
            next_agent = MCP
        else:
            next_agent = _fallback_route(current_step)

        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    """Return the supervisor decision for the conditional graph edge."""

    return state["next_agent"]
