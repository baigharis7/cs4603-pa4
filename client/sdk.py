"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx


class AnalystClientError(Exception):
    """Normalized error raised by the Document Analyst client."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    """Client for a deployed Databricks Document Analyst endpoint."""

    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not endpoint_name.strip():
            raise ValueError("endpoint_name must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        self.endpoint_name = endpoint_name
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.host:
            raise OSError("DATABRICKS_HOST is required")
        if not self.token:
            raise OSError("DATABRICKS_TOKEN is required")

        self._invocations_url = (
            f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        )
        self._status_url = (
            f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
        )
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    @staticmethod
    def _answer_from_payload(payload: Any) -> str:
        """Extract an answer from MLflow graph-state or OpenAI payloads."""

        if isinstance(payload, list) and payload:
            payload = payload[0]

        if not isinstance(payload, dict):
            raise AnalystClientError("Endpoint returned an unsupported response shape")

        final_answer = payload.get("final_answer")
        if final_answer:
            return str(final_answer)

        choices = payload.get("choices") or []
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}
            delta = choice.get("delta") or {}
            content = message.get("content") or delta.get("content")
            if content:
                return str(content)

        messages = payload.get("messages") or []
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict) and last_message.get("content"):
                return str(last_message["content"])

        raise AnalystClientError("Endpoint response did not contain an answer")

    @staticmethod
    def _error_from_response(response: httpx.Response) -> AnalystClientError:
        """Convert an HTTP failure into an AnalystClientError."""

        request_id = response.headers.get("x-request-id") or response.headers.get(
            "x-databricks-request-id"
        )
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error") or response.text
        else:
            message = response.text or response.reason_phrase

        return AnalystClientError(
            str(message),
            status_code=response.status_code,
            request_id=request_id,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send a request with exponential backoff for 429 and 503."""

        started = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TimeoutException as exc:
                elapsed = time.perf_counter() - started
                raise TimeoutError(
                    f"Document Analyst request timed out after {elapsed:.3f} seconds"
                ) from exc
            except httpx.HTTPError as exc:
                raise AnalystClientError(f"Request failed: {exc}") from exc

            if response.status_code in {429, 503} and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue

            if response.is_error:
                raise self._error_from_response(response)

            return response

        raise AnalystClientError("Request failed after retries")

    def ask(self, question: str) -> str:
        """Return a complete answer for a question."""

        if not question.strip():
            raise ValueError("question must not be empty")

        response = self._request(
            "POST",
            self._invocations_url,
            json={"messages": [{"role": "user", "content": question}]},
        )
        return self._answer_from_payload(response.json())

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield SSE text chunks, or one complete answer when streaming is unavailable."""

        if not question.strip():
            raise ValueError("question must not be empty")

        started = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                with self._client.stream(
                    "POST",
                    self._invocations_url,
                    headers={"Accept": "text/event-stream"},
                    json={
                        "messages": [{"role": "user", "content": question}],
                        "stream": True,
                    },
                ) as response:
                    if response.status_code in {429, 503} and attempt < self.max_retries:
                        time.sleep(2**attempt)
                        continue

                    if response.status_code in {400, 404, 405, 422}:
                        yield self.ask(question)
                        return

                    if response.is_error:
                        response.read()
                        raise self._error_from_response(response)

                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" not in content_type:
                        response.read()
                        yield self._answer_from_payload(response.json())
                        return

                    yielded = False
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue

                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        try:
                            chunk = self._answer_from_payload(payload)
                        except AnalystClientError:
                            continue
                        if chunk:
                            yielded = True
                            yield chunk

                    if not yielded:
                        raise AnalystClientError("Streaming response contained no text")
                    return

            except httpx.TimeoutException as exc:
                elapsed = time.perf_counter() - started
                raise TimeoutError(
                    f"Document Analyst stream timed out after {elapsed:.3f} seconds"
                ) from exc
            except httpx.HTTPError as exc:
                raise AnalystClientError(f"Streaming request failed: {exc}") from exc

        raise AnalystClientError("Streaming request failed after retries")

    def health_check(self) -> bool:
        """Return True only when the serving endpoint reports READY."""

        try:
            response = self._request("GET", self._status_url)
            state = response.json().get("state", {})
            return state.get("ready") == "READY"
        except (AnalystClientError, TimeoutError, ValueError):
            return False

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

    def __enter__(self) -> DocumentAnalystClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()