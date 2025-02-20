# How to visualize your graph

Table of contents

- [Setup](https://langchain-ai.github.io/langgraph/how-tos/visualization/#setup)
- [Set up Graph](https://langchain-ai.github.io/langgraph/how-tos/visualization/#set-up-graph)
- [Mermaid](https://langchain-ai.github.io/langgraph/how-tos/visualization/#mermaid)
- [PNG](https://langchain-ai.github.io/langgraph/how-tos/visualization/#png)

  - [Using Mermaid.Ink](https://langchain-ai.github.io/langgraph/how-tos/visualization/#using-mermaidink)
  - [Using Mermaid + Pyppeteer](https://langchain-ai.github.io/langgraph/how-tos/visualization/#using-mermaid-pyppeteer)
  - [Using Graphviz](https://langchain-ai.github.io/langgraph/how-tos/visualization/#using-graphviz)

1.  [Home](https://langchain-ai.github.io/langgraph/)
2.  [Guides](https://langchain-ai.github.io/langgraph/how-tos/)
3.  [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/)
4.  [LangGraph](https://langchain-ai.github.io/langgraph/how-tos#langgraph)
5.  [Graph API Basics](https://langchain-ai.github.io/langgraph/how-tos#graph-api-basics)

# How to visualize your graph[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#how-to-visualize-your-graph "Permanent link")

This guide walks through how to visualize the graphs you create. This works with ANY [Graph](https://langchain-ai.github.io/langgraph/reference/graphs/).

## Setup[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#setup "Permanent link")

First, let's install the required packages

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-0-1)%%capture --no-stderr [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-0-2)%pip install -U langgraph`

## Set up Graph[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#set-up-graph "Permanent link")

You can visualize any arbitrary [Graph](https://langchain-ai.github.io/langgraph/reference/graphs/), including [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph). Let's have some fun by drawing fractals :).

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-1)import random [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-2)from typing import Annotated, Literal [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-3)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-4)from typing_extensions import TypedDict [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-5)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-6)from langgraph.graph import StateGraph, START, END [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-7)from langgraph.graph.message import add_messages [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-8) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-9)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-10)class State(TypedDict):     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-11)    messages: Annotated[list, add_messages] [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-12) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-13)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-14)class MyNode:     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-15)    def __init__(self, name: str):        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-16)        self.name = name [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-17)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-18)    def __call__(self, state: State):        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-19)        return {"messages": [("assistant", f"Called node {self.name}")]} [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-20) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-21)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-22)def route(state) -> Literal["entry_node", "__end__"]:     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-23)    if len(state["messages"]) > 10:        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-24)        return "__end__"    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-25)    return "entry_node" [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-26) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-27)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-28)def add_fractal_nodes(builder, current_node, level, max_level):     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-29)    if level > max_level:        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-30)        return [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-31)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-32)    # Number of nodes to create at this level    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-33)    num_nodes = random.randint(1, 3)  # Adjust randomness as needed    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-34)    for i in range(num_nodes):        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-35)        nm = ["A", "B", "C"][i]        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-36)        node_name = f"node_{current_node}_{nm}"        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-37)        builder.add_node(node_name, MyNode(node_name))        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-38)        builder.add_edge(current_node, node_name) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-39)        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-40)        # Recursively add more nodes        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-41)        r = random.random()        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-42)        if r > 0.2 and level + 1 < max_level:            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-43)            add_fractal_nodes(builder, node_name, level + 1, max_level)        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-44)        elif r > 0.05:            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-45)            builder.add_conditional_edges(node_name, route, node_name)        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-46)        else:            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-47)            # End            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-48)            builder.add_edge(node_name, "__end__") [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-49) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-50)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-51)def build_fractal_graph(max_level: int):     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-52)    builder = StateGraph(State)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-53)    entry_point = "entry_node"    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-54)    builder.add_node(entry_point, MyNode(entry_point))    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-55)    builder.add_edge(START, entry_point) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-56)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-57)    add_fractal_nodes(builder, entry_point, 1, max_level) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-58)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-59)    # Optional: set a finish point if required    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-60)    builder.add_edge(entry_point, END)  # or any specific node [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-61)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-62)    return builder.compile() [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-63) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-64)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-1-65)app = build_fractal_graph(3)`

API Reference: [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph) | [START](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START) | [END](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.END) | [add_messages](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages)

## Mermaid[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#mermaid "Permanent link")

We can also convert a graph class into Mermaid syntax.

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-2-1)print(app.get_graph().draw_mermaid())`

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-1)%%{init: {'flowchart': {'curve': 'linear'}}}%% [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-2)graph TD;     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-3)    __start__([<p>__start__</p>]):::first    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-4)    entry_node(entry_node)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-5)    node_entry_node_A(node_entry_node_A)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-6)    node_entry_node_B(node_entry_node_B)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-7)    node_node_entry_node_B_A(node_node_entry_node_B_A)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-8)    node_node_entry_node_B_B(node_node_entry_node_B_B)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-9)    node_node_entry_node_B_C(node_node_entry_node_B_C)    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-10)    __end__([<p>__end__</p>]):::last    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-11)    __start__ --> entry_node;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-12)    entry_node --> __end__;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-13)    entry_node --> node_entry_node_A;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-14)    entry_node --> node_entry_node_B;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-15)    node_entry_node_B --> node_node_entry_node_B_A;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-16)    node_entry_node_B --> node_node_entry_node_B_B;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-17)    node_entry_node_B --> node_node_entry_node_B_C;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-18)    node_entry_node_A -.-> entry_node;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-19)    node_entry_node_A -.-> __end__;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-20)    node_node_entry_node_B_A -.-> entry_node;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-21)    node_node_entry_node_B_A -.-> __end__;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-22)    node_node_entry_node_B_B -.-> entry_node;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-23)    node_node_entry_node_B_B -.-> __end__;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-24)    node_node_entry_node_B_C -.-> entry_node;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-25)    node_node_entry_node_B_C -.-> __end__;    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-26)    classDef default fill:#f2f0ff,line-height:1.2    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-27)    classDef first fill-opacity:0    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-3-28)    classDef last fill:#bfb6fc`

## PNG[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#png "Permanent link")

If preferred, we could render the Graph into a `.png`. Here we could use three options:

- Using Mermaid.ink API (does not require additional packages)
- Using Mermaid + Pyppeteer (requires `pip install pyppeteer`)
- Using graphviz (which requires `pip install graphviz`)

### Using Mermaid.Ink[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#using-mermaidink "Permanent link")

By default, `draw_mermaid_png()` uses Mermaid.Ink's API to generate the diagram.

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-1)from IPython.display import Image, display [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-2)from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-3)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-4)display(     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-5)    Image(        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-6)        app.get_graph().draw_mermaid_png(            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-7)            draw_method=MermaidDrawMethod.API,        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-8)        )    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-9)    ) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-4-10))`

API Reference: [CurveStyle](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.graph.CurveStyle.html) | [MermaidDrawMethod](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.graph.MermaidDrawMethod.html) | [NodeStyles](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.graph.NodeStyles.html)

The visual of the graph will be rendered as a visual and the code of it would look something like this:

```mermaid
flowchart TD
    start(["__start__"]) --> entry_node[entry_node]

    entry_node --> node_entry_node_A[node_entry_node_A]
    entry_node --> node_entry_node_B[node_entry_node_B]

    node_entry_node_B --> node_node_entry_node_B_A[node_node_entry_node_B_A]
    node_entry_node_B --> node_node_entry_node_B_B[node_node_entry_node_B_B]
    node_entry_node_B --> node_node_entry_node_B_C[node_node_entry_node_B_C]

    node_entry_node_A -.-> entry_node
    node_node_entry_node_B_A -.-> entry_node
    node_node_entry_node_B_B -.-> entry_node
    node_node_entry_node_B_C -.-> entry_node

    node_entry_node_A --> finish(["__end__"])
    node_node_entry_node_B_A -.-> finish
    node_node_entry_node_B_B -.-> finish
    node_node_entry_node_B_C -.-> finish

    style start fill:#b19cd9,stroke:#9370db
    style finish fill:#b19cd9,stroke:#9370db
    style entry_node fill:#e6e6fa,stroke:#9370db
    style node_entry_node_A fill:#e6e6fa,stroke:#9370db
    style node_entry_node_B fill:#e6e6fa,stroke:#9370db
    style node_node_entry_node_B_A fill:#e6e6fa,stroke:#9370db
    style node_node_entry_node_B_B fill:#e6e6fa,stroke:#9370db
    style node_node_entry_node_B_C fill:#e6e6fa,stroke:#9370db
```

### Using Mermaid + Pyppeteer[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#using-mermaid-pyppeteer "Permanent link")

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-5-1)%%capture --no-stderr [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-5-2)%pip install --quiet pyppeteer [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-5-3)%pip install --quiet nest_asyncio`

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-1)import nest_asyncio [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-2)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-3)nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-4)[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-5)display(     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-6)    Image(        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-7)        app.get_graph().draw_mermaid_png(            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-8)            curve_style=CurveStyle.LINEAR,            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-9)            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-10)            wrap_label_n_words=9,            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-11)            output_file_path=None,            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-12)            draw_method=MermaidDrawMethod.PYPPETEER,            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-13)            background_color="white",            [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-14)            padding=10,        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-15)        )    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-16)    ) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-6-17))`

### Using Graphviz[¶](https://langchain-ai.github.io/langgraph/how-tos/visualization/#using-graphviz "Permanent link")

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-7-1)%%capture --no-stderr [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-7-2)%pip install pygraphviz`

`[](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-8-1)try:     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-8-2)    display(Image(app.get_graph().draw_png())) [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-8-3)except ImportError:     [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-8-4)    print(        [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-8-5)        "You likely need to install dependencies for pygraphviz, see more here https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"    [](https://langchain-ai.github.io/langgraph/how-tos/visualization/#__codelineno-8-6)    )`
