"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store():
    """Connect to the managed Databricks Vector Search index."""

    from databricks_langchain import DatabricksVectorSearch

    settings = get_settings()

    if not settings["vs_endpoint"] or not settings["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX must be configured"
        )

    return DatabricksVectorSearch(
        index_name=settings["vs_index"],
        endpoint=settings["vs_endpoint"],
        # text_column=TEXT_COLUMN,
        primary_key="chunk_id",
        doc_uri="source",
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    """Return a top-k LangChain retriever over the managed index."""

    if k < 1:
        raise ValueError("k must be at least 1")

    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})