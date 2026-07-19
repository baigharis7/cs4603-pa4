"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

PIP_REQUIREMENTS = [
    "mlflow>=2.16.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "databricks-langchain>=0.1.0",
    "databricks-vectorsearch>=0.40",
    "databricks-sdk>=0.23.0",
    "mcp>=1.0.0",
    "langchain-mcp-adapters>=0.0.5",
    "openai>=1.40.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
]


def _require(name: str) -> str:
    """Read a required deployment setting."""

    value = os.environ.get(name)
    if not value:
        raise OSError(f"Missing required environment variable: {name}")
    return value


@contextmanager
def _portable_model_code_path():
    """Keep MLflow models-from-code paths portable when packaging on Windows."""

    from mlflow.langchain.utils import logging as logging_utils

    original = logging_utils._validate_and_get_model_code_path

    def validate(path: str, temp_dir: str) -> str:
        return original(path, temp_dir).replace("\\", "/")

    logging_utils._validate_and_get_model_code_path = validate
    try:
        yield
    finally:
        logging_utils._validate_and_get_model_code_path = original


def log_and_register() -> tuple[str, str]:
    """Log the graph with MLflow and register it in Unity Catalog."""

    catalog = _require("UC_CATALOG")
    schema = _require("UC_SCHEMA")

    model_name = os.environ.get(
        "REGISTERED_MODEL_NAME",
        "haris_document_analyst",
    )
    uc_name = f"{catalog}.{schema}.{model_name}"

    workspace = WorkspaceClient()
    username = workspace.current_user.me().user_name
    experiment_name = f"/Users/{username}/cs4603-pa4-document-analyst"

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="document-analyst-deployment"):
        with _portable_model_code_path():
            model_info = mlflow.langchain.log_model(
                lc_model="deployment/agent_model.py",
                name="agent",
                code_paths=[
                    str(ROOT / "agent"),
                    str(ROOT / "rag"),
                    str(ROOT / "tools"),
                    str(ROOT / "config.py"),
                ],
                pip_requirements=PIP_REQUIREMENTS,
                input_example={
                    "messages": [
                        {
                            "role": "user",
                            "content": "What was Meridian's FY2023 net revenue?",
                        }
                    ]
                },
            )

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=uc_name,
    )

    version = str(registered.version)

    print(f"Registered model: {uc_name}")
    print(f"Registered version: {version}")

    return uc_name, version


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create or update the Document Analyst serving endpoint."""

    endpoint_name = _require("SERVING_ENDPOINT_NAME")
    secret_scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")

    host = _require("DATABRICKS_HOST").rstrip("/")
    vs_endpoint = _require("VECTOR_SEARCH_ENDPOINT")
    vs_index = _require("VECTOR_SEARCH_INDEX")
    embeddings_endpoint = os.environ.get(
        "EMBEDDINGS_ENDPOINT",
        "databricks-gte-large-en",
    )

    environment_vars = {
        "DATABRICKS_HOST": (
            f"{{{{secrets/{secret_scope}/DATABRICKS_HOST}}}}"
        ),
        "DATABRICKS_TOKEN": (
            f"{{{{secrets/{secret_scope}/DATABRICKS_TOKEN}}}}"
        ),
        "DATABRICKS_MODEL": (
            f"{{{{secrets/{secret_scope}/DATABRICKS_MODEL}}}}"
        ),
        "VECTOR_SEARCH_ENDPOINT": vs_endpoint,
        "VECTOR_SEARCH_INDEX": vs_index,
        "EMBEDDINGS_ENDPOINT": embeddings_endpoint,
    }

    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=environment_vars,
    )

    workspace = WorkspaceClient()
    timeout = timedelta(minutes=30)

    try:
        workspace.serving_endpoints.get(endpoint_name)
    except NotFound:
        print(f"Creating endpoint: {endpoint_name}")

        workspace.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[served_entity],
            ),
            timeout=timeout,
        )
    else:
        print(f"Updating endpoint: {endpoint_name}")

        workspace.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[served_entity],
            timeout=timeout,
        )

    endpoint_url = f"{host}/ml/endpoints/{endpoint_name}"

    print(f"Endpoint READY: {endpoint_name}")
    print(f"Endpoint URL: {endpoint_url}")

    return endpoint_url


if __name__ == "__main__":
    name, version = log_and_register()
    create_or_update_endpoint(name, version)
