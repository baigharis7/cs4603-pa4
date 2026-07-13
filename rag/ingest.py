"""Corpus ingestion into Databricks Vector Search."""

from __future__ import annotations

import os
import time

from config import get_settings


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse documents, prepare semantic chunks, and save a Delta table."""

    spark.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW pa4_parsed_docs AS
        SELECT
            path AS source,
            ai_parse_document(content) AS parsed
        FROM READ_FILES(
            '{volume_path}',
            format => 'binaryFile'
        )
        """
    )

    spark.sql(
        """
        CREATE OR REPLACE TEMP VIEW pa4_prepared_docs AS
        SELECT
            source,
            ai_prep_search(parsed) AS prepared
        FROM pa4_parsed_docs
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {chunks_table}
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        AS
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,
            chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
            chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
            element_at(split(d.source, '/'), -1) AS source,
            chunk.value:pages[0].page_id::INT + 1 AS page
        FROM pa4_prepared_docs AS d,
        LATERAL variant_explode(d.prepared:document.contents) AS chunk
        WHERE chunk.value:chunk_id IS NOT NULL
        """
    )


def create_index():
    """Create and wait for the Standard endpoint and Delta Sync index."""

    from databricks.vector_search.client import VectorSearchClient

    settings = get_settings()
    source_table = os.environ["SOURCE_TABLE"]
    endpoint_name = settings["vs_endpoint"]
    index_name = settings["vs_index"]

    client = VectorSearchClient(
        workspace_url=settings["host"],
        personal_access_token=settings["token"],
        disable_notice=True,
    )

    try:
        client.get_endpoint(endpoint_name)
    except Exception:
        client.create_endpoint(
            name=endpoint_name,
            endpoint_type="STANDARD",
        )

    for _ in range(45):
        endpoint = client.get_endpoint(endpoint_name)
        if endpoint.get("endpoint_status", {}).get("state") == "ONLINE":
            break
        time.sleep(20)
    else:
        raise TimeoutError("Vector Search endpoint did not become ONLINE")

    try:
        index = client.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
        )
    except Exception:
        index = client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=source_table,
            index_name=index_name,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=settings["embeddings"],
        )

    for _ in range(45):
        status = index.describe().get("status", {})
        if status.get("ready"):
            return index
        time.sleep(20)

    raise TimeoutError("Vector Search index did not become READY")