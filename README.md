# CS4603 PA4 — Document Analyst

> This `README.md` is a **graded deliverable**:
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Code that runs but is not explained will not receive full marks.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment PDF.

## Setup

```bash
# Install the locked project environment
uv sync

# Create the local configuration file (PowerShell)
Copy-Item .env.example .env

# Authenticate the Databricks CLI
uv run databricks auth login \
  --host https://dbc-358be7a6-cc24.cloud.databricks.com \
  --profile MLLOPs-demo
```

The local `.env` contains the Databricks host, model endpoint, embedding
endpoint, Vector Search endpoint and index, Unity Catalog location, serving
endpoint name, and secret scope. The access token is kept only in `.env`, which
is excluded by Git. This workspace does not provide the `main` catalog, so the
project uses `workspace.default` instead.

The report was uploaded to
`/Volumes/workspace/default/pa4/annual_report.pdf`. The ingestion notebook
created seven chunks in `workspace.default.haris_analyst_chunks`. The Vector
Search index is `workspace.default.haris_analyst_index` on the
`haris-vs-endpoint` endpoint.

## Running locally

1. Upload `data/annual_report.pdf` to
   `/Volumes/workspace/default/pa4/annual_report.pdf`.
2. Open `pa4.ipynb` in the Databricks workspace and attach it to serverless
   compute. The first cells install the repository as an editable package and
   restart Python:

```python
%pip install -q -e /Workspace/Users/m.harisbaig7@gmail.com/cs4603-pa4
dbutils.library.restartPython()
```

3. Run the Part 0 cells. They parse the PDF with `ai_parse_document`, prepare
   it with `ai_prep_search`, write the seven chunks to
   `workspace.default.haris_analyst_chunks`, and create or reuse the Vector
   Search resources. Wait until `workspace.default.haris_analyst_index` reports
   `ONLINE` and `ready=True` before testing retrieval.
4. Run the Part 1 cells to import and compile the graph:

```python
from agent.graph import build_graph

graph = build_graph()
print("Graph compiled successfully")
```

5. From the local repository, run the automated checks:

```bash
uv run ruff check agent client deployment rag config.py
uv run pytest -q
```

The graph was tested with the three required query types:

```text
Retrieval-only:
What was Meridian's net income in FY2023?
Result: ¥1,107 billion [source: annual_report.pdf, p.2]

Computation-only:
What is 15% of 2.4 billion?
Result: 360 million

Combined retrieval and computation:
If Meridian's FY2023 net revenue grows at 8% annually for 3 years,
what will it be?
Result: ¥16,910 billion retrieved, projected to ¥21,301.7 billion
[source: annual_report.pdf, p.2]
```

The combined trace showed one retrieval step followed by one MCP calculation
step before synthesis. This confirms that the supervisor routed each operation
to the appropriate specialist.

## Deployment

The model is logged with MLflow models-from-code and registered in Unity
Catalog by running:

```bash
uv run python deployment/deploy.py
```

`deployment/agent_model.py` rebuilds the production graph from the packaged
source, connects it to Databricks Model Serving and Vector Search, loads the MCP
tools, and calls `mlflow.models.set_model(graph)`. The deployment code packages
the required project directories and Python dependencies. It supplies runtime
credentials to the endpoint through Databricks secret references rather than
putting secret values in the model artifact.

The deployed resources used for the final test were:

```text
Registered model: workspace.default.haris_document_analyst
Registered version: 5
Serving endpoint: haris-document-analyst
Endpoint URL: https://dbc-358be7a6-cc24.cloud.databricks.com/ml/endpoints/haris-document-analyst
```

The endpoint reached `READY`. A live request returned the correct cited net
income, and all three deployed test queries matched their local results. The
first request took 38.90 seconds while the immediate warm request took 3.73
seconds. The difference is consistent with scale-to-zero startup and model
initialization.

## Design decisions

- `AnalystState` keeps the plan, current step, intermediate results, routing
  decision, messages, and final answer in one typed state shared by every node.
- The planner separates retrieval from calculation so a specialist only
  receives the work it is designed to perform.
- The supervisor routes document questions to the RAG agent, arithmetic to the
  deterministic MCP tools, and completed plans to the synthesizer.
- Databricks Vector Search stores the report externally. This allows the index
  to be refreshed without registering another model version.
- MCP tools perform calculations instead of asking the language model to do
  arithmetic. This makes the numeric step repeatable and easier to test.
- Retrieved answers retain the source filename and page number, and the
  synthesizer preserves those citations in the final answer.
- The SDK handles the endpoint's graph-state response, retries temporary
  failures with exponential backoff, supports a timeout, and exposes a streaming
  interface. The deployed endpoint currently returns one complete chunk, which
  is still a valid stream for this response format.

---

## Analysis Questions

### Task 1.2 — Planner

1. **What happens when the planner produces steps that depend on each other (e.g., step 3 needs the result of step 1)? How does your architecture handle this?**

   Steps are executed in plan order. `current_step_index` identifies the next
   step, while `step_results` keeps the output of every completed step. Later
   nodes receive those earlier results as context. For the combined test, the
   RAG agent first retrieved FY2023 net revenue of ¥16,910 billion. The MCP step
   then used 16,910 as the starting value for the three-year 8% compound-growth
   calculation, producing ¥21,301.7 billion.

2. **Would a replanning step after each execution improve or hurt performance for this use case? Justify with an example.**

   A fixed plan is faster, cheaper, and more predictable for short questions
   because it avoids another language-model call after each result. Re-planning
   is useful when retrieval fails, an unexpected unit appears, or a result
   changes what should happen next, but it adds latency and can make a correct
   plan less stable. For this assignment, a fixed plan is appropriate. A
   production version could re-plan only when a node reports a failure or an
   unusable result.

### Task 1.3 — Supervisor

1. **Your supervisor makes a routing decision per step. What is the failure mode if it misroutes? How would you detect and recover from a misroute?**

   If a retrieval step goes to MCP, no document fact is available for a valid
   calculation. If a calculation goes to RAG, the retriever may return related
   text but will not reliably compute the requested value. The system can detect
   this through empty retrieval results, a missing tool call, an invalid numeric
   result, or an exception recorded in the trace. A production recovery path
   would retry the step with the other specialist or ask the planner for a
   revised step. In the current graph, failures are made explicit instead of
   being silently presented as a successful result.

2. **Compare this supervisor pattern with a single ReAct agent that has access to all tools. When is the supervisor pattern worth the added complexity?**

   The supervisor is worthwhile when a question combines different kinds of
   work that need separate tools, prompts, and validation rules. It gives a
   visible execution path and keeps arithmetic deterministic. A single ReAct
   agent is simpler for one-step or open-ended questions, but its tool choices
   and calculations are less constrained. The financial-report workflow
   benefits from a supervisor because retrieval, calculation, and synthesis
   have clearly different responsibilities.

### Task 1.4 — RAG Agent

1. **The RAG agent retrieves for a single decomposed step, not the full user query. How does this affect retrieval quality compared to retrieving for the original question?**

   The main advantage is a focused search query. Calculation instructions and
   unrelated parts of the original request do not weaken the embedding match.
   The disadvantage is that an overly short step can lose useful context such
   as the company, fiscal year, metric, or unit. The planner therefore needs to
   keep those details in every retrieval step. For example, “Retrieve Meridian's
   FY2023 net revenue” is safer than “Find the revenue.”

2. **If the planner produces a vague step like “find relevant financial data,” how would you improve the retrieval query before sending it to the vector store?**

   Query rewriting can turn vague wording into the language used in the report.
   It can add the entity, year, metric, and likely synonyms before searching. A
   request for “sales last year,” for example, can be rewritten as “Meridian
   Motor Corporation FY2023 net revenue financial highlights.” Multi-query
   retrieval could search two or three such variants and merge the results, but
   it would increase latency and Vector Search calls.

### Task 2.1 — Model Definition

1. **Why does `models-from-code` require a self-contained file? What breaks if you reference external state, such as a database running only on your laptop?**

   Model Serving loads the registered artifact in a clean Linux container. It
   cannot use the repository working directory, Windows paths, local virtual
   environment, or files that were not packaged. Depending on them causes import
   errors, missing-file errors, or client initialization failures during model
   loading. This project packages its source and requirements, resolves the MCP
   server from the model artifact, and obtains service configuration from
   endpoint environment variables.

2. **What are the trade-offs in freshness, cold-start size, latency, and failure modes between querying an external Vector Search index and baking the corpus into the model artifact?**

   External Vector Search keeps the model artifact smaller and allows documents
   and embeddings to be refreshed without registering a new model. It also
   provides managed indexing and filtering. Its costs are an additional network
   call, authentication requirements, and dependence on another service. Baking
   documents into the artifact removes that remote dependency and can reduce
   retrieval latency for a tiny fixed collection, but it makes the artifact
   larger and stale until the model is packaged and deployed again.

### Task 2.3 — Serving Endpoint

1. **Why must you pass `DATABRICKS_TOKEN` as an environment variable to the endpoint, even though it is already authenticated to serve models?**

   Authentication for an incoming endpoint request does not automatically give
   the model permission to make outgoing calls. This graph creates clients that
   call the language-model endpoint and Vector Search, so those clients still
   need workspace credentials. The token is supplied through a Databricks secret
   reference, which prevents the actual value from being stored in source code
   or the registered model.

2. **What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?**

   Databricks prepares the new endpoint configuration while the active
   configuration continues serving traffic. Once the new model is ready, traffic
   is moved to it; requests already accepted by the old configuration can finish
   there. If an update fails and a healthy active configuration exists, the old
   version remains available. During this project's first failed deployment
   there was no healthy active configuration yet, so the endpoint remained
   `NOT_READY` until a corrected model version was deployed.

### Task 3.2 — Client

1. **Why is exponential backoff better than fixed-interval retries for a model serving endpoint?**

   Exponential backoff retries quickly after a brief failure but spaces later
   attempts farther apart. This gives a scaled-to-zero endpoint or rate-limited
   service time to recover and avoids many clients retrying at the same fixed
   interval. The client retries temporary HTTP and connection failures but does
   not hide permanent errors indefinitely.

2. **What is the danger of setting `max_retries` too high in a production system with many concurrent users?**

   A caller can wait too long for a failure that will not recover. Repeated
   requests also consume client connections, endpoint capacity, and quota, and
   can increase pressure during an outage. Retries should therefore be bounded
   by both an attempt count and a request timeout. A larger production system
   should also use jitter and a circuit breaker.

3. **When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.**

   Streaming is useful when an answer is long and the user should see progress
   before generation finishes, such as a detailed explanation of several years
   of financial performance. `ask()` is better for short answers, scripts, and
   batch jobs where the caller needs one complete value. The current serving
   response is returned as one graph-state result, so the SDK exposes it as one
   complete stream chunk; the interface can support finer-grained chunks if the
   endpoint later provides token streaming.

### Bonus A / B / C (if attempted)

The optional bonus tasks were not attempted. Parts 0 through 3, including the
required client streaming interface, were completed.