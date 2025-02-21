# Dependencies - PydanticAI

Table of contents

- [Defining Dependencies](#defining-dependencies)
- [Accessing Dependencies](#accessing-dependencies)

  - [Asynchronous vs. Synchronous dependencies](#asynchronous-vs-synchronous-dependencies)

- [Full Example](#full-example)
- [Overriding Dependencies](#overriding-dependencies)
- [Examples](#examples)

1.  [Introduction](..)
2.  [Documentation](../agents/)

Version Notice

This documentation is ahead of the last release by [15 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.24...main). You may see documentation for features not yet supported in the latest release [v0.0.24 2025-02-12](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.24).

# Dependencies

PydanticAI uses a dependency injection system to provide data and services to your agent's [system prompts](../agents/#system-prompts), [tools](../tools/) and [result validators](../results/#result-validators-functions).

Matching PydanticAI's design philosophy, our dependency system tries to use existing best practice in Python development rather than inventing esoteric "magic", this should make dependencies type-safe, understandable easier to test and ultimately easier to deploy in production.

## Defining Dependencies

Dependencies can be any python type. While in simple cases you might be able to pass a single object as a dependency (e.g. an HTTP connection), [dataclasses](https://docs.python.org/3/library/dataclasses.html#module-dataclasses) are generally a convenient container when your dependencies included multiple objects.

Here's an example of defining an agent that requires dependencies.

(**Note:** dependencies aren't actually used in this example, see [Accessing Dependencies](#accessing-dependencies) below)

unused_dependencies.py

`from dataclasses import dataclass  import httpx  from pydantic_ai import Agent  @dataclass class MyDeps:    [](#__code_0_annotation_1)      api_key: str     http_client: httpx.AsyncClient  agent = Agent(     'openai:gpt-4o',     deps_type=MyDeps,    [](#__code_0_annotation_2)  )  async def main():     async with httpx.AsyncClient() as client:         deps = MyDeps('foobar', client)         result = await agent.run(             'Tell me a joke.',             deps=deps,    [](#__code_0_annotation_3)          )         print(result.data)         #> Did you hear about the toothpaste scandal? They called it Colgate.`

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Accessing Dependencies

Dependencies are accessed through the [`RunContext`](../api/tools/#pydantic_ai.tools.RunContext) type, this should be the first parameter of system prompt functions etc.

system_prompt_dependencies.py

`from dataclasses import dataclass  import httpx  from pydantic_ai import Agent, RunContext  @dataclass class MyDeps:     api_key: str     http_client: httpx.AsyncClient  agent = Agent(     'openai:gpt-4o',     deps_type=MyDeps, )  @agent.system_prompt    [](#__code_1_annotation_1)   async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:    [](#__code_1_annotation_2)      response = await ctx.deps.http_client.get(    [](#__code_1_annotation_3)        'https://example.com',        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},    [](#__code_1_annotation_4)    )    response.raise_for_status()    return f'Prompt: {response.text}'  async def main():     async with httpx.AsyncClient() as client:         deps = MyDeps('foobar', client)         result = await agent.run('Tell me a joke.', deps=deps)         print(result.data)         #> Did you hear about the toothpaste scandal? They called it Colgate.`

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

### Asynchronous vs. Synchronous dependencies

[System prompt functions](../agents/#system-prompts), [function tools](../tools/) and [result validators](../results/#result-validators-functions) are all run in the async context of an agent run.

If these functions are not coroutines (e.g. `async def`) they are called with [`run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) in a thread pool, it's therefore marginally preferable to use `async` methods where dependencies perform IO, although synchronous dependencies should work fine too.

`run` vs. `run_sync` and Asynchronous vs. Synchronous dependencies

Whether you use synchronous or asynchronous dependencies, is completely independent of whether you use `run` or `run_sync` — `run_sync` is just a wrapper around `run` and agents are always run in an async context.

Here's the same example as above, but with a synchronous dependency:

sync_dependencies.py

`from dataclasses import dataclass  import httpx  from pydantic_ai import Agent, RunContext  @dataclass class MyDeps:     api_key: str     http_client: httpx.Client    [](#__code_2_annotation_1)  agent = Agent(     'openai:gpt-4o',     deps_type=MyDeps, )  @agent.system_prompt def get_system_prompt(ctx: RunContext[MyDeps]) -> str:    [](#__code_2_annotation_2)      response = ctx.deps.http_client.get(         'https://example.com', headers={'Authorization': f'Bearer {ctx.deps.api_key}'}     )     response.raise_for_status()     return f'Prompt: {response.text}'  async def main():     deps = MyDeps('foobar', httpx.Client())     result = await agent.run(         'Tell me a joke.',         deps=deps,     )     print(result.data)     #> Did you hear about the toothpaste scandal? They called it Colgate.`

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Full Example

As well as system prompts, dependencies can be used in [tools](../tools/) and [result validators](../results/#result-validators-functions).

full_example.py

`from dataclasses import dataclass  import httpx  from pydantic_ai import Agent, ModelRetry, RunContext  @dataclass class MyDeps:     api_key: str     http_client: httpx.AsyncClient  agent = Agent(     'openai:gpt-4o',     deps_type=MyDeps, )  @agent.system_prompt async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:     response = await ctx.deps.http_client.get('https://example.com')     response.raise_for_status()     return f'Prompt: {response.text}'  @agent.tool    [](#__code_3_annotation_1)   async def get_joke_material(ctx: RunContext[MyDeps], subject: str) -> str:     response = await ctx.deps.http_client.get(        'https://example.com#jokes',        params={'subject': subject},        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},    )    response.raise_for_status()    return response.text  @agent.result_validator    [](#__code_3_annotation_2)   async def validate_result(ctx: RunContext[MyDeps], final_response: str) -> str:     response = await ctx.deps.http_client.post(        'https://example.com#validate',        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},        params={'query': final_response},    )    if response.status_code == 400:        raise ModelRetry(f'invalid response: {response.text}')    response.raise_for_status()    return final_response  async def main():     async with httpx.AsyncClient() as client:         deps = MyDeps('foobar', client)         result = await agent.run('Tell me a joke.', deps=deps)         print(result.data)         #> Did you hear about the toothpaste scandal? They called it Colgate.`

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Overriding Dependencies

When testing agents, it's useful to be able to customise dependencies.

While this can sometimes be done by calling the agent directly within unit tests, we can also override dependencies while calling application code which in turn calls the agent.

This is done via the [`override`](../api/agent/#pydantic_ai.agent.Agent.override) method on the agent.

joke_app.py

`from dataclasses import dataclass  import httpx  from pydantic_ai import Agent, RunContext  @dataclass class MyDeps:     api_key: str     http_client: httpx.AsyncClient      async def system_prompt_factory(self) -> str:    [](#__code_4_annotation_1)          response = await self.http_client.get('https://example.com')         response.raise_for_status()         return f'Prompt: {response.text}'  joke_agent = Agent('openai:gpt-4o', deps_type=MyDeps)  @joke_agent.system_prompt async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:     return await ctx.deps.system_prompt_factory()    [](#__code_4_annotation_2)  async def application_code(prompt: str) -> str:    [](#__code_4_annotation_3)      ...     ...     # now deep within application code we call our agent     async with httpx.AsyncClient() as client:         app_deps = MyDeps('foobar', client)         result = await joke_agent.run(prompt, deps=app_deps)    [](#__code_4_annotation_4)      return result.data`

_(This example is complete, it can be run "as is")_

test_joke_app.py

`from joke_app import MyDeps, application_code, joke_agent  class TestMyDeps(MyDeps):    [](#__code_5_annotation_1)      async def system_prompt_factory(self) -> str:         return 'test prompt'  async def test_application_code():     test_deps = TestMyDeps('test_key', None)    [](#__code_5_annotation_2)    with joke_agent.override(deps=test_deps):    [](#__code_5_annotation_3)        joke = await application_code('Tell me a joke.')    [](#__code_5_annotation_4)    assert joke.startswith('Did you hear about the toothpaste scandal?')`

## Examples

The following examples demonstrate how to use dependencies in PydanticAI:

- [Weather Agent](../../examples/weather_agent.py)
- [SQL Generation](../../examples/sql_generation.py)
- [RAG](../../examples/rag.py)
