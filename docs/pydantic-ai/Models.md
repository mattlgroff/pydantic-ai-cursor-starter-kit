# Models - PydanticAI

Table of contents

- [OpenAI](#openai)

  - [Install](#install)
  - [Configuration](#configuration)
  - [Environment variable](#environment-variable)
  - [api_key argument](#api_key-argument)
  - [Custom OpenAI Client](#custom-openai-client)

- [Anthropic](#anthropic)

  - [Install](#install_1)
  - [Configuration](#configuration_1)
  - [Environment variable](#environment-variable_1)
  - [api_key argument](#api_key-argument_1)

- [Gemini](#gemini)

  - [Install](#install_2)
  - [Configuration](#configuration_2)
  - [Environment variable](#environment-variable_2)
  - [api_key argument](#api_key-argument_2)

- [Gemini via VertexAI](#gemini-via-vertexai)

  - [Install](#install_3)
  - [Configuration](#configuration_3)
  - [Application default credentials](#application-default-credentials)
  - [Service account](#service-account)
  - [Customising region](#customising-region)

- [Groq](#groq)

  - [Install](#install_4)
  - [Configuration](#configuration_4)
  - [Environment variable](#environment-variable_3)
  - [api_key argument](#api_key-argument_3)

- [Mistral](#mistral)

  - [Install](#install_5)
  - [Configuration](#configuration_5)
  - [Environment variable](#environment-variable_4)
  - [api_key argument](#api_key-argument_4)

- [Cohere](#cohere)

  - [Install](#install_6)
  - [Configuration](#configuration_6)
  - [Environment variable](#environment-variable_5)
  - [api_key argument](#api_key-argument_5)

- [OpenAI-compatible Models](#openai-compatible-models)

  - [Ollama](#ollama)

    - [Example local usage](#example-local-usage)
    - [Example using a remote server](#example-using-a-remote-server)

  - [OpenRouter](#openrouter)
  - [Grok (xAI)](#grok-xai)
  - [DeepSeek](#deepseek)
  - [Perplexity](#perplexity)

- [Implementing Custom Models](#implementing-custom-models)

1.  [Introduction](..)
2.  [Documentation](../agents/)

Version Notice

This documentation is ahead of the last release by [15 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.24...main). You may see documentation for features not yet supported in the latest release [v0.0.24 2025-02-12](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.24).

# Models

PydanticAI is Model-agnostic and has built in support for the following model providers:

- [OpenAI](#openai)
- [Anthropic](#anthropic)
- Gemini via two different APIs: [Generative Language API](#gemini) and [VertexAI API](#gemini-via-vertexai)
- [Ollama](#ollama)
- [Deepseek](#deepseek)
- [Groq](#groq)
- [Mistral](#mistral)
- [Cohere](#cohere)

See [OpenAI-compatible models](#openai-compatible-models) for more examples on how to use models such as [OpenRouter](#openrouter), and [Grok (xAI)](#grok-xai) that support the OpenAI SDK.

You can also [add support for other models](#implementing-custom-models).

PydanticAI also comes with [`TestModel`](../api/models/test/) and [`FunctionModel`](../api/models/function/) for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right packages installed.

## OpenAI

### Install

To use OpenAI models, you need to either install [`pydantic-ai`](../install/), or install [`pydantic-ai-slim`](../install/#slim-install) with the `openai` optional group:

[pip](#__tabbed_1_1)[uv](#__tabbed_1_2)

`pip install 'pydantic-ai-slim[openai]'`

`uv add 'pydantic-ai-slim[openai]'`

### Configuration

To use [`OpenAIModel`](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel) through their main API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key.

### Environment variable

Once you have the API key, you can set it as an environment variable:

`export OPENAI_API_KEY='your-api-key'`

You can then use [`OpenAIModel`](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel) by name:

openai_model_by_name.py

`from pydantic_ai import Agent  agent = Agent('openai:gpt-4o') ...`

Or initialise the model directly with just the model name:

openai_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  model = OpenAIModel('gpt-4o') agent = Agent(model) ...`

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__):

openai_model_api_key.py

`from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  model = OpenAIModel('gpt-4o', api_key='your-api-key') agent = Agent(model) ...`

### Custom OpenAI Client

`OpenAIModel` also accepts a custom `AsyncOpenAI` client via the [`openai_client` parameter](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__), so you can customise the `organization`, `project`, `base_url` etc. as defined in the [OpenAI API docs](https://platform.openai.com/docs/api-reference).

You could also use the [`AsyncAzureOpenAI`](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints) client to use the Azure OpenAI API.

openai_azure.py

`from openai import AsyncAzureOpenAI  from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  client = AsyncAzureOpenAI(     azure_endpoint='...',     api_version='2024-07-01-preview',     api_key='your-api-key', )  model = OpenAIModel('gpt-4o', openai_client=client) agent = Agent(model) ...`

## Anthropic

### Install

To use [`AnthropicModel`](../api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel) models, you need to either install [`pydantic-ai`](../install/), or install [`pydantic-ai-slim`](../install/#slim-install) with the `anthropic` optional group:

[pip](#__tabbed_2_1)[uv](#__tabbed_2_2)

`pip install 'pydantic-ai-slim[anthropic]'`

`uv add 'pydantic-ai-slim[anthropic]'`

### Configuration

To use [Anthropic](https://anthropic.com) through their API, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) to generate an API key.

[`AnthropicModelName`](../api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModelName) contains a list of available Anthropic models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

`export ANTHROPIC_API_KEY='your-api-key'`

You can then use [`AnthropicModel`](../api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel) by name:

anthropic_model_by_name.py

`from pydantic_ai import Agent  agent = Agent('anthropic:claude-3-5-sonnet-latest') ...`

Or initialise the model directly with just the model name:

anthropic_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.anthropic import AnthropicModel  model = AnthropicModel('claude-3-5-sonnet-latest') agent = Agent(model) ...`

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](../api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel.__init__):

anthropic_model_api_key.py

`from pydantic_ai import Agent from pydantic_ai.models.anthropic import AnthropicModel  model = AnthropicModel('claude-3-5-sonnet-latest', api_key='your-api-key') agent = Agent(model) ...`

## Gemini

For prototyping only

Google themselves refer to this API as the "hobby" API, I've received 503 responses from it a number of times. The API is easy to use and useful for prototyping and simple demos, but I would not rely on it in production.

If you want to run Gemini models in production, you should use the [VertexAI API](#gemini-via-vertexai) described below.

### Install

To use [`GeminiModel`](../api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) models, you just need to install [`pydantic-ai`](../install/) or [`pydantic-ai-slim`](../install/#slim-install), no extra dependencies are required.

### Configuration

[`GeminiModel`](../api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) let's you use the Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods), `generativelanguage.googleapis.com`.

[`GeminiModelName`](../api/models/gemini/#pydantic_ai.models.gemini.GeminiModelName) contains a list of available Gemini models that can be used through this interface.

To use `GeminiModel`, go to [aistudio.google.com](https://aistudio.google.com/) and follow your nose until you find the place to generate an API key.

### Environment variable

Once you have the API key, you can set it as an environment variable:

`export GEMINI_API_KEY=your-api-key`

You can then use [`GeminiModel`](../api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) by name:

gemini_model_by_name.py

`from pydantic_ai import Agent  agent = Agent('google-gla:gemini-1.5-flash') ...`

Note

The `google-gla` provider prefix represents the [Google **G**enerative **L**anguage **A**PI](https://ai.google.dev/api/all-methods) for `GeminiModel`s. `google-vertex` is used with [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) for `VertexAIModel`s.

Or initialise the model directly with just the model name:

gemini_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.gemini import GeminiModel  model = GeminiModel('gemini-1.5-flash') agent = Agent(model) ...`

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](../api/models/gemini/#pydantic_ai.models.gemini.GeminiModel.__init__):

gemini_model_api_key.py

`from pydantic_ai import Agent from pydantic_ai.models.gemini import GeminiModel  model = GeminiModel('gemini-1.5-flash', api_key='your-api-key') agent = Agent(model) ...`

## Gemini via VertexAI

To run Google's Gemini models in production, you should use [`VertexAIModel`](../api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel) which uses the `*-aiplatform.googleapis.com` API.

[`GeminiModelName`](../api/models/gemini/#pydantic_ai.models.gemini.GeminiModelName) contains a list of available Gemini models that can be used through this interface.

### Install

To use [`VertexAIModel`](../api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel), you need to either install [`pydantic-ai`](../install/), or install [`pydantic-ai-slim`](../install/#slim-install) with the `vertexai` optional group:

[pip](#__tabbed_3_1)[uv](#__tabbed_3_2)

`pip install 'pydantic-ai-slim[vertexai]'`

`uv add 'pydantic-ai-slim[vertexai]'`

### Configuration

This interface has a number of advantages over `generativelanguage.googleapis.com` documented above:

1.  The VertexAI API is more reliably and marginally lower latency in our experience.
2.  You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with VertexAI to guarantee capacity.
3.  If you're running PydanticAI inside GCP, you don't need to set up authentication, it should "just work".
4.  You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

The big disadvantage is that for local development you may need to create and configure a "service account", which I've found extremely painful to get right in the past.

Whichever way you authenticate, you'll need to have VertexAI enabled in your GCP account.

### Application default credentials

Luckily if you're running PydanticAI inside GCP, or you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you should be able to use `VertexAIModel` without any additional setup.

To use `VertexAIModel`, with [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) configured (e.g. with `gcloud`), you can simply use:

vertexai_application_default_credentials.py

`from pydantic_ai import Agent from pydantic_ai.models.vertexai import VertexAIModel  model = VertexAIModel('gemini-1.5-flash') agent = Agent(model) ...`

Internally this uses [`google.auth.default()`](https://google-auth.readthedocs.io/en/master/reference/google.auth.html) from the `google-auth` package to obtain credentials.

Won't fail until `agent.run()`

Because `google.auth.default()` requires network requests and can be slow, it's not run until you call `agent.run()`. Meaning any configuration or permissions error will only be raised when you try to use the model. To initialize the model for this check to be run, call [`await model.ainit()`](../api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel.ainit).

You may also need to pass the [`project_id` argument to `VertexAIModel`](../api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel.__init__) if application default credentials don't set a project, if you pass `project_id` and it conflicts with the project set by application default credentials, an error is raised.

### Service account

If instead of application default credentials, you want to authenticate with a service account, you'll need to create a service account, add it to your GCP project (note: AFAIK this step is necessary even if you created the service account within the project), give that service account the "Vertex AI Service Agent" role, and download the service account JSON file.

Once you have the JSON file, you can use it thus:

vertexai_service_account.py

`from pydantic_ai import Agent from pydantic_ai.models.vertexai import VertexAIModel  model = VertexAIModel(     'gemini-1.5-flash',     service_account_file='path/to/service-account.json', ) agent = Agent(model) ...`

### Customising region

Whichever way you authenticate, you can specify which region requests will be sent to via the [`region` argument](../api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel.__init__).

Using a region close to your application can improve latency and might be important from a regulatory perspective.

vertexai_region.py

`from pydantic_ai import Agent from pydantic_ai.models.vertexai import VertexAIModel  model = VertexAIModel('gemini-1.5-flash', region='asia-east1') agent = Agent(model) ...`

[`VertexAiRegion`](../api/models/vertexai/#pydantic_ai.models.vertexai.VertexAiRegion) contains a list of available regions.

## Groq

### Install

To use [`GroqModel`](../api/models/groq/#pydantic_ai.models.groq.GroqModel), you need to either install [`pydantic-ai`](../install/), or install [`pydantic-ai-slim`](../install/#slim-install) with the `groq` optional group:

[pip](#__tabbed_4_1)[uv](#__tabbed_4_2)

`pip install 'pydantic-ai-slim[groq]'`

`uv add 'pydantic-ai-slim[groq]'`

### Configuration

To use [Groq](https://groq.com/) through their API, go to [console.groq.com/keys](https://console.groq.com/keys) and follow your nose until you find the place to generate an API key.

[`GroqModelName`](../api/models/groq/#pydantic_ai.models.groq.GroqModelName) contains a list of available Groq models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

`export GROQ_API_KEY='your-api-key'`

You can then use [`GroqModel`](../api/models/groq/#pydantic_ai.models.groq.GroqModel) by name:

groq_model_by_name.py

`from pydantic_ai import Agent  agent = Agent('groq:llama-3.3-70b-versatile') ...`

Or initialise the model directly with just the model name:

groq_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.groq import GroqModel  model = GroqModel('llama-3.3-70b-versatile') agent = Agent(model) ...`

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](../api/models/groq/#pydantic_ai.models.groq.GroqModel.__init__):

groq_model_api_key.py

`from pydantic_ai import Agent from pydantic_ai.models.groq import GroqModel  model = GroqModel('llama-3.3-70b-versatile', api_key='your-api-key') agent = Agent(model) ...`

## Mistral

### Install

To use [`MistralModel`](../api/models/mistral/#pydantic_ai.models.mistral.MistralModel), you need to either install [`pydantic-ai`](../install/), or install [`pydantic-ai-slim`](../install/#slim-install) with the `mistral` optional group:

[pip](#__tabbed_5_1)[uv](#__tabbed_5_2)

`pip install 'pydantic-ai-slim[mistral]'`

`uv add 'pydantic-ai-slim[mistral]'`

### Configuration

To use [Mistral](https://mistral.ai) through their API, go to [console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/) and follow your nose until you find the place to generate an API key.

[`MistralModelName`](../api/models/mistral/#pydantic_ai.models.mistral.MistralModelName) contains a list of the most popular Mistral models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

`export MISTRAL_API_KEY='your-api-key'`

You can then use [`MistralModel`](../api/models/mistral/#pydantic_ai.models.mistral.MistralModel) by name:

mistral_model_by_name.py

`from pydantic_ai import Agent  agent = Agent('mistral:mistral-large-latest') ...`

Or initialise the model directly with just the model name:

mistral_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.mistral import MistralModel  model = MistralModel('mistral-small-latest') agent = Agent(model) ...`

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](../api/models/mistral/#pydantic_ai.models.mistral.MistralModel.__init__):

mistral_model_api_key.py

`from pydantic_ai import Agent from pydantic_ai.models.mistral import MistralModel  model = MistralModel('mistral-small-latest', api_key='your-api-key') agent = Agent(model) ...`

## Cohere

### Install

To use [`CohereModel`](../api/models/cohere/#pydantic_ai.models.cohere.CohereModel), you need to either install [`pydantic-ai`](../install/), or install [`pydantic-ai-slim`](../install/#slim-install) with the `cohere` optional group:

[pip](#__tabbed_6_1)[uv](#__tabbed_6_2)

`pip install 'pydantic-ai-slim[cohere]'`

`uv add 'pydantic-ai-slim[cohere]'`

### Configuration

To use [Cohere](https://cohere.com/) through their API, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key.

[`CohereModelName`](../api/models/cohere/#pydantic_ai.models.cohere.CohereModelName) contains a list of the most popular Cohere models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

`export CO_API_KEY='your-api-key'`

You can then use [`CohereModel`](../api/models/cohere/#pydantic_ai.models.cohere.CohereModel) by name:

cohere_model_by_name.py

`from pydantic_ai import Agent  agent = Agent('cohere:command') ...`

Or initialise the model directly with just the model name:

cohere_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.cohere import CohereModel  model = CohereModel('command', api_key='your-api-key') agent = Agent(model) ...`

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](../api/models/cohere/#pydantic_ai.models.cohere.CohereModel.__init__):

cohere_model_api_key.py

`from pydantic_ai import Agent from pydantic_ai.models.cohere import CohereModel  model = CohereModel('command', api_key='your-api-key') agent = Agent(model) ...`

## OpenAI-compatible Models

Many of the models are compatible with OpenAI API, and thus can be used with [`OpenAIModel`](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel) in PydanticAI. Before getting started, check the [OpenAI](#openai) section for installation and configuration instructions.

To use another OpenAI-compatible API, you can make use of the [`base_url`](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__) and [`api_key`](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__) arguments:

openai_model_base_url.py

`from pydantic_ai.models.openai import OpenAIModel  model = OpenAIModel(     'model_name',     base_url='https://<openai-compatible-api-endpoint>.com',    api_key='your-api-key', ) ...`

### Ollama

To use [Ollama](https://ollama.com/), you must first download the Ollama client, and then download a model using the [Ollama model library](https://ollama.com/library).

You must also ensure the Ollama server is running when trying to make requests to it. For more information, please see the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs).

#### Example local usage

With `ollama` installed, you can run the server with the model you want to use:

terminal-run-ollama

`ollama run llama3.2`

(this will pull the `llama3.2` model if you don't already have it downloaded)

Then run your code, here's a minimal example:

ollama_example.py

`from pydantic import BaseModel  from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  class CityLocation(BaseModel):     city: str     country: str  ollama_model = OpenAIModel(model_name='llama3.2', base_url='http://localhost:11434/v1') agent = Agent(ollama_model, result_type=CityLocation)  result = agent.run_sync('Where were the olympics held in 2012?') print(result.data) #> city='London' country='United Kingdom' print(result.usage()) """ Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None) """`

#### Example using a remote server

ollama_example_with_remote_server.py

`from pydantic import BaseModel  from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  ollama_model = OpenAIModel(     model_name='qwen2.5-coder:7b',    [](#__code_43_annotation_1)      base_url='http://192.168.1.74:11434/v1',    [](#__code_43_annotation_2)  )  class CityLocation(BaseModel):     city: str     country: str  agent = Agent(model=ollama_model, result_type=CityLocation)  result = agent.run_sync('Where were the olympics held in 2012?') print(result.data) #> city='London' country='United Kingdom' print(result.usage()) """ Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None) """`

### OpenRouter

To use [OpenRouter](https://openrouter.ai), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

Once you have the API key, you can pass it to [`OpenAIModel`](../api/models/openai/#pydantic_ai.models.openai.OpenAIModel) as the `api_key` argument:

openrouter_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  model = OpenAIModel(     'anthropic/claude-3.5-sonnet',     base_url='https://openrouter.ai/api/v1',     api_key='your-openrouter-api-key', ) agent = Agent(model) ...`

### Grok (xAI)

Go to [xAI API Console](https://console.x.ai/) and create an API key. Once you have the API key, follow the [xAI API Documentation](https://docs.x.ai/docs/overview), and set the `base_url` and `api_key` arguments appropriately:

grok_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  model = OpenAIModel(     'grok-2-1212',     base_url='https://api.x.ai/v1',     api_key='your-xai-api-key', ) agent = Agent(model) ...`

### DeepSeek

Go to [DeepSeek API Platform](https://platform.deepseek.com/api_keys) and create an API key. Once you have the API key, follow the [DeepSeek API Documentation](https://platform.deepseek.com/docs/api/overview), and set the `base_url` and `api_key` arguments appropriately:

deepseek_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  model = OpenAIModel(     'deepseek-chat',     base_url='https://api.deepseek.com',     api_key='your-deepseek-api-key', ) agent = Agent(model) ...`

### Perplexity

Follow the Perplexity [getting started](https://docs.perplexity.ai/guides/getting-started) guide to create an API key. Then, you can query the Perplexity API with the following:

perplexity_model_init.py

`from pydantic_ai import Agent from pydantic_ai.models.openai import OpenAIModel  model = OpenAIModel(     'sonar-pro',     base_url='https://api.perplexity.ai',     api_key='your-perplexity-api-key', ) agent = Agent(model) ...`

## Implementing Custom Models

To implement support for models not already supported, you will need to subclass the [`Model`](../api/models/base/#pydantic_ai.models.Model) abstract base class.

For streaming, you'll also need to implement the following abstract base class:

- [`StreamedResponse`](../api/models/base/#pydantic_ai.models.StreamedResponse)

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to PydanticAI, see the [contributing guidelines](../contributing/#new-model-rules).

© Pydantic Services Inc. 2024 to present
