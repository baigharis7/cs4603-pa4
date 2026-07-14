"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _message_text(message) -> str:
    """Extract text from a LangChain message or dictionary."""

    if hasattr(message, "content"):
        return str(message.content)

    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(message)


def make_synthesizer(llm):
    """Create the final synthesis node."""

    def synthesizer(state: AnalystState) -> dict:
        original_question = _message_text(state["messages"][0])
        results = state.get("step_results", [])

        numbered_results = "\n".join(
            f"Step {position}: {result}"
            for position, result in enumerate(results, start=1)
        )

        if not numbered_results:
            numbered_results = "No step results were produced."

        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(
                    content=(
                        f"Original question:\n{original_question}\n\n"
                        f"Completed step results:\n{numbered_results}"
                    )
                ),
            ]
        )

        final_answer = str(response.content).strip()

        if not final_answer:
            final_answer = "Unable to synthesize an answer from the available results."

        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }

    return synthesizer