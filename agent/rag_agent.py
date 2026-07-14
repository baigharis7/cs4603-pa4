"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def _format_page(page) -> str:
    """Format numeric page values without a trailing decimal."""

    if isinstance(page, float) and page.is_integer():
        return str(int(page))
    return str(page)


def format_docs(docs) -> str:
    """Format retrieved documents with source citations."""

    formatted = []

    for doc in docs:
        metadata = doc.metadata
        source = metadata.get("source") or metadata.get("doc_uri", "unknown")
        page = _format_page(metadata.get("page", "?"))
        citation = f"[source: {source}, p.{page}]"
        formatted.append(f"{citation}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(formatted)


def make_rag_agent(retriever, llm):
    """Create the RAG specialist node."""

    def rag_agent(state: AnalystState) -> dict:
        index = state["current_step_index"]
        current_step = state["plan"][index]
        docs = retriever.invoke(current_step)

        if not docs:
            result = "not found in documents"
        else:
            context = format_docs(docs)
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(
                        content=(
                            f"Current retrieval step:\n{current_step}\n\n"
                            f"Document chunks:\n{context}"
                        )
                    ),
                ]
            )
            result = str(response.content).strip() or "not found in documents"

        return {
            "step_results": [*state.get("step_results", []), result],
            "current_step_index": index + 1,
        }

    return rag_agent