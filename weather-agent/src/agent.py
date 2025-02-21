import asyncio
from typing import Any

from devtools import debug
from pydantic_ai import Agent, RunContext


weather_agent = Agent(
    "groq:llama-3.3-70b-versatile",
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        "Be concise, reply with one sentence."
        "Use the `get_weather` tool to get the weather in a city"
    ),
    deps_type=None,
    retries=2,
)


@weather_agent.tool
async def get_weather(ctx: RunContext[None], city_name: str) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        city_name: The name of the city.
    """

    if city_name == "London":
        return {
            "temperature": "100°C",
            "description": "Boiling hot.",
        }

    return {
        "temperature": "0°C",
        "description": "Freezing cold.",
    }


async def main():
    result = await weather_agent.run(
        "What is the weather like in London and in Wiltshire?"
    )
    debug(result)
    print("Response:", result.data)


if __name__ == "__main__":
    asyncio.run(main())
