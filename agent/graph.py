"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""


from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _run_async(operation):
    """Run an async operation from normal code, including inside notebooks."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(operation())).result()


def _configure_mcp_stdio() -> None:
    """Give MCP subprocesses a real stderr file in serving containers."""

    from langchain_mcp_adapters import sessions

    if getattr(sessions, "_pa4_stdio_configured", False):
        return

    original_stdio_client = sessions.stdio_client

    @asynccontextmanager
    async def stdio_client_with_real_stderr(server_params):
        with open(os.devnull, "w", encoding="utf-8") as errlog:
            async with original_stdio_client(
                server_params,
                errlog=errlog,
            ) as streams:
                yield streams

    sessions.stdio_client = stdio_client_with_real_stderr
    sessions._pa4_stdio_configured = True


def load_mcp_tools(server_path: str | None = None):
    """Load the provided MCP tools once through the stdio transport."""

    _configure_mcp_stdio()

    from langchain_mcp_adapters.client import MultiServerMCPClient

    if server_path is None:
        server_path = str(
            Path(__file__).resolve().parents[1] / "tools" / "mcp_server.py"
        )

    client = MultiServerMCPClient(
        {
            "analyst": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [server_path],
            }
        }
    )

    return _run_async(client.get_tools)




def _tool_result_text(value) -> str:
    """Extract plain text from MCP content blocks."""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        texts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                texts.append(str(item.text))
            else:
                texts.append(str(item))
        return "\n".join(text for text in texts if text)

    return str(value)


def make_mcp_node(tools, llm):
    """Create the deterministic calculation specialist node."""

    tool_map = {tool.name: tool for tool in tools}
    tool_calling_llm = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        index = state["current_step_index"]
        current_step = state["plan"][index]
        previous_results = state.get("step_results", [])

        previous_context = "\n".join(
            f"Step {position}: {result}"
            for position, result in enumerate(previous_results, start=1)
        )
        if not previous_context:
            previous_context = "No previous step results."

        response = tool_calling_llm.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(
                    content=(
                        f"Current calculation step:\n{current_step}\n\n"
                        f"Previous results:\n{previous_context}"
                    )
                ),
            ]
        )

        tool_calls = getattr(response, "tool_calls", [])

        if not tool_calls:
            result = "Calculation failed: the model did not call an MCP tool."
        else:
            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            arguments = tool_call.get("args", {})
            tool = tool_map.get(tool_name)

            if tool is None:
                result = f"Calculation failed: unknown tool '{tool_name}'."
            else:
                try:
                    result = _tool_result_text(tool.invoke(arguments))
                except NotImplementedError:
                    result = _tool_result_text(
                      _run_async(lambda: tool.ainvoke(arguments))
                      )
                except Exception as exc:
                    result = f"Calculation failed: {exc}"

        return {
            "step_results": [*previous_results, result],
            "current_step_index": index + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Build and compile the complete Document Analyst graph."""

    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm()

    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever()

    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)

    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node(RAG, make_rag_agent(retriever, llm))
    builder.add_node(MCP, make_mcp_node(tools, llm))
    builder.add_node(SYNTH, make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            RAG: RAG,
            MCP: MCP,
            SYNTH: SYNTH,
        },
    )
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()
