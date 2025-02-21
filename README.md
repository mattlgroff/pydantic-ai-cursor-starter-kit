# Pydantic AI

This is a starter kit for building an agent using Pydantic AI.

## Documentation

The following documentation is available in `./docs/pydantic-ai/`:

- **Core Concepts**

  - [Agents](./docs/pydantic-ai/Agents.md)
  - [Models](./docs/pydantic-ai/Models.md)
  - [Dependencies](./docs/pydantic-ai/Dependencies.md)
  - [Function Tools](./docs/pydantic-ai/Function_Tools.md)
  - [Structured Result Validation](./docs/pydantic-ai/Structured_Result_Validation.md)
  - [Messages and Chat History](./docs/pydantic-ai/Messages_and_chat_history.md)

- **Advanced Topics**
  - [Testing and Evals](./docs/pydantic-ai/Testing_and_Evals.md)
  - [Multi-agent Applications](./docs/pydantic-ai/Multi_agent_Applications.md)
  - [Graphs](./docs/pydantic-ai/Graphs.md)

For the complete documentation, visit [ai.pydantic.dev](https://ai.pydantic.dev/).

## Examples

The following examples are available in `./examples/`:

- `pydantic_model.py` - Basic Pydantic model usage
- `weather_agent.py` - Simple weather agent implementation
- `bank_support.py` - Bank support agent with dependency injection
- `sql_generation.py` - SQL query generation
- `multi_agent_flow_flight_booking.py` - Multi-agent flight booking system
- `rag.py` - Retrieval Augmented Generation example
- `stream_markdown.py` - Streaming markdown content
- `stream_structured_objects.py` - Streaming structured data
- `chat_app_with_fastapi.py` - FastAPI chat application integration

## Run the Weather Agent Example

```bash
cd weather-agent

uv venv

source .venv/bin/activate

uv sync

uv run src/agent.py
```
