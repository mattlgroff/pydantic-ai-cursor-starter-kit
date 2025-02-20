# Application Structure

Table of contents

- [Overview](https://langchain-ai.github.io/langgraph/concepts/application_structure/#overview)
- [Key Concepts](https://langchain-ai.github.io/langgraph/concepts/application_structure/#key-concepts)
- [File Structure](https://langchain-ai.github.io/langgraph/concepts/application_structure/#file-structure)
- [Configuration File](https://langchain-ai.github.io/langgraph/concepts/application_structure/#configuration-file)

  - [Examples](https://langchain-ai.github.io/langgraph/concepts/application_structure/#examples)

- [Dependencies](https://langchain-ai.github.io/langgraph/concepts/application_structure/#dependencies)
- [Graphs](https://langchain-ai.github.io/langgraph/concepts/application_structure/#graphs)
- [Environment Variables](https://langchain-ai.github.io/langgraph/concepts/application_structure/#environment-variables)
- [Related](https://langchain-ai.github.io/langgraph/concepts/application_structure/#related)

1.  [Home](https://langchain-ai.github.io/langgraph/)
2.  [Guides](https://langchain-ai.github.io/langgraph/how-tos/)
3.  [Concepts](https://langchain-ai.github.io/langgraph/concepts/)
4.  [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts#langgraph-platform)
5.  [LangGraph Server](https://langchain-ai.github.io/langgraph/concepts#langgraph-server)

[](https://github.com/langchain-ai/langgraph/edit/main/docs/docs/concepts/application_structure.md "Edit this page")

# Application Structure[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#application-structure "Permanent link")

Prerequisites

- [LangGraph Server](./LangGraph_Server_.md)
- [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/)

## Overview[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#overview "Permanent link")

A LangGraph application consists of one or more graphs, a LangGraph API Configuration file (`langgraph.json`), a file that specifies dependencies, and an optional .env file that specifies environment variables.

This guide shows a typical structure for a LangGraph application and shows how the required information to deploy a LangGraph application using the LangGraph Platform is specified.

## Key Concepts[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#key-concepts "Permanent link")

To deploy using the LangGraph Platform, the following information should be provided:

1.  A [LangGraph API Configuration file](https://langchain-ai.github.io/langgraph/concepts/application_structure/#configuration-file) (`langgraph.json`) that specifies the dependencies, graphs, environment variables to use for the application.
2.  The [graphs](https://langchain-ai.github.io/langgraph/concepts/application_structure/#graphs) that implement the logic of the application.
3.  A file that specifies [dependencies](https://langchain-ai.github.io/langgraph/concepts/application_structure/#dependencies) required to run the application.
4.  [Environment variable](https://langchain-ai.github.io/langgraph/concepts/application_structure/#environment-variables) that are required for the application to run.

## File Structure[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#file-structure "Permanent link")

Below are examples of directory structures for Python and JavaScript applications:

[Python (requirements.txt)](#__tabbed_1_1)[Python (pyproject.toml)](#__tabbed_1_2)[JS (package.json)](#__tabbed_1_3)

`[](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-1)my-app/ [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-2)├── my_agent # all project code lies within here [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-3)│   ├── utils # utilities for your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-4)│   │   ├── __init__.py [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-5)│   │   ├── tools.py # tools for your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-6)│   │   ├── nodes.py # node functions for you graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-7)│   │   └── state.py # state definition of your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-8)│   ├── __init__.py [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-9)│   └── agent.py # code for constructing your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-10)├── .env # environment variables [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-11)├── requirements.txt # package dependencies [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-0-12)└── langgraph.json # configuration file for LangGraph`

`[](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-1)my-app/ [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-2)├── my_agent # all project code lies within here [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-3)│   ├── utils # utilities for your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-4)│   │   ├── __init__.py [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-5)│   │   ├── tools.py # tools for your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-6)│   │   ├── nodes.py # node functions for you graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-7)│   │   └── state.py # state definition of your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-8)│   ├── __init__.py [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-9)│   └── agent.py # code for constructing your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-10)├── .env # environment variables [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-11)├── langgraph.json  # configuration file for LangGraph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-1-12)└── pyproject.toml # dependencies for your project`

`[](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-1)my-app/ [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-2)├── src # all project code lies within here [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-3)│   ├── utils # optional utilities for your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-4)│   │   ├── tools.ts # tools for your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-5)│   │   ├── nodes.ts # node functions for you graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-6)│   │   └── state.ts # state definition of your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-7)│   └── agent.ts # code for constructing your graph [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-8)├── package.json # package dependencies [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-9)├── .env # environment variables [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-2-10)└── langgraph.json # configuration file for LangGraph`

Note

The directory structure of a LangGraph application can vary depending on the programming language and the package manager used.

## Configuration File[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#configuration-file "Permanent link")

The `langgraph.json` file is a JSON file that specifies the dependencies, graphs, environment variables, and other settings required to deploy a LangGraph application.

The file supports specification of the following information:

Key

Description

`dependencies`

**Required**. Array of dependencies for LangGraph API server. Dependencies can be one of the following: (1) `"."`, which will look for local Python packages, (2) `pyproject.toml`, `setup.py` or `requirements.txt` in the app directory `"./local_package"`, or (3) a package name.

`graphs`

**Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example:

- `./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`
- `./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and creates an instance of `langgraph.graph.state.StateGraph` / `langgraph.graph.state.CompiledStateGraph`.

`env`

Path to `.env` file or a mapping from environment variable to its value.

`python_version`

`3.11` or `3.12`. Defaults to `3.11`.

`pip_config_file`

Path to `pip` config file.

`dockerfile_lines`

Array of additional lines to add to Dockerfile following the import from parent image.

Tip

The LangGraph CLI defaults to using the configuration file **langgraph.json** in the current directory.

### Examples[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#examples "Permanent link")

[Python](#__tabbed_2_1)[JavaScript](#__tabbed_2_2)

- The dependencies involve a custom local package and the `langchain_openai` package.
- A single graph will be loaded from the file `./your_package/your_file.py` with the variable `variable`.
- The environment variables are loaded from the `.env` file.

`[](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-1){     [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-2)    "dependencies": [        [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-3)        "langchain_openai",        [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-4)        "./your_package"    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-5)    ],    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-6)    "graphs": {        [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-7)        "my_agent": "./your_package/your_file.py:agent"    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-8)    },    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-9)    "env": "./.env" [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-3-10)}`

- The dependencies will be loaded from a dependency file in the local directory (e.g., `package.json`).
- A single graph will be loaded from the file `./your_package/your_file.js` with the function `agent`.
- The environment variable `OPENAI_API_KEY` is set inline.

`[](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-1){     [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-2)    "dependencies": [        [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-3)        "."    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-4)    ],    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-5)    "graphs": {        [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-6)        "my_agent": "./your_package/your_file.js:agent"    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-7)    },    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-8)    "env": {        [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-9)        "OPENAI_API_KEY": "secret-key"    [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-10)    } [](https://langchain-ai.github.io/langgraph/concepts/application_structure/#__codelineno-4-11)}`

## Dependencies[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#dependencies "Permanent link")

A LangGraph application may depend on other Python packages or JavaScript libraries (depending on the programming language in which the application is written).

You will generally need to specify the following information for dependencies to be set up correctly:

1.  A file in the directory that specifies the dependencies (e.g., `requirements.txt`, `pyproject.toml`, or `package.json`).
2.  A `dependencies` key in the [LangGraph configuration file](https://langchain-ai.github.io/langgraph/concepts/application_structure/#configuration-file) that specifies the dependencies required to run the LangGraph application.
3.  Any additional binaries or system libraries can be specified using `dockerfile_lines` key in the [LangGraph configuration file](https://langchain-ai.github.io/langgraph/concepts/application_structure/#configuration-file).

## Graphs[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#graphs "Permanent link")

Use the `graphs` key in the [LangGraph configuration file](https://langchain-ai.github.io/langgraph/concepts/application_structure/#configuration-file) to specify which graphs will be available in the deployed LangGraph application.

You can specify one or more graphs in the configuration file. Each graph is identified by a name (which should be unique) and a path for either: (1) the compiled graph or (2) a function that makes a graph is defined.

## Environment Variables[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#environment-variables "Permanent link")

If you're working with a deployed LangGraph application locally, you can configure environment variables in the `env` key of the [LangGraph configuration file](https://langchain-ai.github.io/langgraph/concepts/application_structure/#configuration-file).

For a production deployment, you will typically want to configure the environment variables in the deployment environment.

## Related[¶](https://langchain-ai.github.io/langgraph/concepts/application_structure/#related "Permanent link")

Please see the following resources for more information:

- How-to guides for [Application Structure](https://langchain-ai.github.io/langgraph/how-tos/#application-structure).
