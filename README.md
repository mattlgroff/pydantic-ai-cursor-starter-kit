# Pydantic AI and LangGraph Agent Starter Kit

This is a starter kit for building an agent using Pydantic AI and LangGraph.

To read more about:

- Pydantic AI read our [local docs here](./docs/pydantic-ai/pydantic-ai-docs.md) or the [latest online docs for LLMs](https://ai.pydantic.dev/llms.txt) or for [humans](https://ai.pydantic.dev)
- LangGraph read our [local docs here](./docs/lang-graph/_LangGraph_Quickstart_.md) or the [latest online docs here](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

## Prerequisites

- Docker

## Spin up LangGraph Server + Pydantic AI Application

```bash
docker compose up
```

This will start `postgres` and `redis` containers and the `langgraph-server` and `pydantic-ai-application` containers.
