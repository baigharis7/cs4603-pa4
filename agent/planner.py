"""Planner node (Task 1.2)."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _message_text(message) -> str:
    """Extract text from either a LangChain message or dictionary."""

    if hasattr(message, "content"):
        return str(message.content)

    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(message)


def _parse_plan(raw_output: str, fallback: str) -> list[str]:
    """Parse a JSON plan, falling back safely when output is malformed."""

    text = raw_output.strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end < start:
        return [fallback]

    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return [fallback]

    if not isinstance(parsed, list):
        return [fallback]

    steps = [step.strip() for step in parsed if isinstance(step, str) and step.strip()]
    return steps[:5] or [fallback]


def make_planner(llm):
    """Create the planner node with an injected language model."""

    def planner(state: AnalystState) -> dict:
        question = _message_text(state["messages"][-1])

        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=question),
            ]
        )

        plan = _parse_plan(str(response.content), fallback=question)

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner