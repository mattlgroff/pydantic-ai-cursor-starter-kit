# Pydantic AI Cursor Starter Kit

This is a starter kit for building an agent using Pydantic AI using [Cursor IDE](https://www.cursor.com/).

It's not a traditional starter kit in that it doesn't have a lot of bells and whistles. It has a lot of documentation, examples, .cursorrules file, and a single example agent implementation.

## Documentation

The following Pydantic AIdocumentation is available in `./docs/`:

- **Core Concepts**

  - [Agents](./docs/Agents.md)
  - [Models](./docs/Models.md)
  - [Dependencies](./docs/Dependencies.md)
  - [Function Tools](./docs/Function_Tools.md)
  - [Structured Result Validation](./docs/Structured_Result_Validation.md)
  - [Messages and Chat History](./docs/Messages_and_chat_history.md)

- **Advanced Topics**
  - [Testing and Evals](./docs/Testing_and_Evals.md)
  - [Multi-agent Applications](./docs/Multi_agent_Applications.md)
  - [Graphs](./docs/Graphs.md)

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
