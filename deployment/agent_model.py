"""
TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""


from __future__ import annotations

from pathlib import Path

import mlflow

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm, get_settings
from rag.store import get_retriever

# Validate required Databricks credentials when MLflow loads the model.
get_settings()

# Resolve the bundled MCP server independently of the working directory.
_MODEL_DIR = Path(__file__).resolve().parent

# Local execution places tools beside deployment/.
# MLflow places code_paths inside the model's code/ directory.
_MCP_CANDIDATES = [
    _MODEL_DIR.parent / "tools" / "mcp_server.py",
    _MODEL_DIR / "code" / "tools" / "mcp_server.py",
]

_MCP_SERVER = next(
    (path for path in _MCP_CANDIDATES if path.is_file()),
    None,
)

if _MCP_SERVER is None:
    searched = ", ".join(str(path) for path in _MCP_CANDIDATES)
    raise FileNotFoundError(f"MCP server not found. Searched: {searched}")

# Construct production dependencies once during model loading.
llm = get_chat_llm()
retriever = get_retriever()


tools = load_mcp_tools(str(_MCP_SERVER))

graph = build_graph(
    llm=llm,
    retriever=retriever,
    tools=tools,
)

mlflow.models.set_model(graph)
