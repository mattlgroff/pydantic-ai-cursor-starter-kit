# LangGraph Glossary

Table of contents

- [Graphs](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs)

  - [StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph)
  - [Compiling your graph](https://langchain-ai.github.io/langgraph/concepts/low_level/#compiling-your-graph)

- [State](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)

  - [Schema](https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)

    - [Multiple schemas](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas)

  - [Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)

    - [Default Reducer](https://langchain-ai.github.io/langgraph/concepts/low_level/#default-reducer)

  - [Working with Messages in Graph State](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state)

    - [Why use messages?](https://langchain-ai.github.io/langgraph/concepts/low_level/#why-use-messages)
    - [Using Messages in your Graph](https://langchain-ai.github.io/langgraph/concepts/low_level/#using-messages-in-your-graph)
    - [Serialization](https://langchain-ai.github.io/langgraph/concepts/low_level/#serialization)
    - [MessagesState](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate)

- [Nodes](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes)

  - [START Node](https://langchain-ai.github.io/langgraph/concepts/low_level/#start-node)
  - [END Node](https://langchain-ai.github.io/langgraph/concepts/low_level/#end-node)

- [Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges)

  - [Normal Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#normal-edges)
  - [Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
  - [Entry Point](https://langchain-ai.github.io/langgraph/concepts/low_level/#entry-point)
  - [Conditional Entry Point](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-entry-point)

- [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [Command](https://langchain-ai.github.io/langgraph/concepts/low_level/#command)

  - [When should I use Command instead of conditional edges?](https://langchain-ai.github.io/langgraph/concepts/low_level/#when-should-i-use-command-instead-of-conditional-edges)
  - [Navigating to a node in a parent graph](https://langchain-ai.github.io/langgraph/concepts/low_level/#navigating-to-a-node-in-a-parent-graph)
  - [Using inside tools](https://langchain-ai.github.io/langgraph/concepts/low_level/#using-inside-tools)
  - [Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/low_level/#human-in-the-loop)

- [Persistence](https://langchain-ai.github.io/langgraph/concepts/low_level/#persistence)
- [Threads](https://langchain-ai.github.io/langgraph/concepts/low_level/#threads)
- [Storage](https://langchain-ai.github.io/langgraph/concepts/low_level/#storage)
- [Graph Migrations](https://langchain-ai.github.io/langgraph/concepts/low_level/#graph-migrations)
- [Configuration](https://langchain-ai.github.io/langgraph/concepts/low_level/#configuration)

  - [Recursion Limit](https://langchain-ai.github.io/langgraph/concepts/low_level/#recursion-limit)

- [interrupt](https://langchain-ai.github.io/langgraph/concepts/low_level/#interrupt)
- [Breakpoints](https://langchain-ai.github.io/langgraph/concepts/low_level/#breakpoints)
- [Subgraphs](https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs)

  - [As a compiled graph](https://langchain-ai.github.io/langgraph/concepts/low_level/#as-a-compiled-graph)
  - [As a function](https://langchain-ai.github.io/langgraph/concepts/low_level/#as-a-function)

- [Visualization](https://langchain-ai.github.io/langgraph/concepts/low_level/#visualization)
- [Streaming](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming)

1.  [Home](https://langchain-ai.github.io/langgraph/)
2.  [Guides](https://langchain-ai.github.io/langgraph/how-tos/)
3.  [Concepts](https://langchain-ai.github.io/langgraph/concepts/)
4.  [LangGraph](https://langchain-ai.github.io/langgraph/concepts#langgraph)

# LangGraph Glossary[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#langgraph-glossary "Permanent link")

## Graphs[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs "Permanent link")

At its core, LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components:

1.  [`State`](https://langchain-ai.github.io/langgraph/concepts/low_level/#state): A shared data structure that represents the current snapshot of your application. It can be any Python type, but is typically a `TypedDict` or Pydantic `BaseModel`.
2.  [`Nodes`](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes): Python functions that encode the logic of your agents. They receive the current `State` as input, perform some computation or side-effect, and return an updated `State`.
3.  [`Edges`](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges): Python functions that determine which `Node` to execute next based on the current `State`. They can be conditional branches or fixed transitions.

By composing `Nodes` and `Edges`, you can create complex, looping workflows that evolve the `State` over time. The real power, though, comes from how LangGraph manages that `State`. To emphasize: `Nodes` and `Edges` are nothing more than Python functions - they can contain an LLM or just good ol' Python code.

In short: _nodes do the work. edges tell what to do next_.

LangGraph's underlying graph algorithm uses [message passing](https://en.wikipedia.org/wiki/Message_passing) to define a general program. When a Node completes its operation, it sends messages along one or more edges to other node(s). These recipient nodes then execute their functions, pass the resulting messages to the next set of nodes, and the process continues. Inspired by Google's [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) system, the program proceeds in discrete "super-steps."

A super-step can be considered a single iteration over the graph nodes. Nodes that run in parallel are part of the same super-step, while nodes that run sequentially belong to separate super-steps. At the start of graph execution, all nodes begin in an `inactive` state. A node becomes `active` when it receives a new message (state) on any of its incoming edges (or "channels"). The active node then runs its function and responds with updates. At the end of each super-step, nodes with no incoming messages vote to `halt` by marking themselves as `inactive`. The graph execution terminates when all nodes are `inactive` and no messages are in transit.

### StateGraph[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph "Permanent link")

The `StateGraph` class is the main graph class to use. This is parameterized by a user defined `State` object.

### Compiling your graph[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#compiling-your-graph "Permanent link")

To build your graph, you first define the [state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state), you then add [nodes](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes) and [edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges), and then you compile it. What exactly is compiling your graph and why is it needed?

Compiling is a pretty simple step. It provides a few basic checks on the structure of your graph (no orphaned nodes, etc). It is also where you can specify runtime args like [checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/) and [breakpoints](https://langchain-ai.github.io/langgraph/concepts/low_level/#breakpoints). You compile your graph by just calling the `.compile` method:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-0-1)graph = graph_builder.compile(...)`

You **MUST** compile your graph before you can use it.

## State[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#state "Permanent link")

The first thing you do when you define a graph is define the `State` of the graph. The `State` consists of the [schema of the graph](https://langchain-ai.github.io/langgraph/concepts/low_level/#schema) as well as [`reducer` functions](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) which specify how to apply updates to the state. The schema of the `State` will be the input schema to all `Nodes` and `Edges` in the graph, and can be either a `TypedDict` or a `Pydantic` model. All `Nodes` will emit updates to the `State` which are then applied using the specified `reducer` function.

### Schema[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#schema "Permanent link")

The main documented way to specify the schema of a graph is by using `TypedDict`. However, we also support [using a Pydantic BaseModel](https://langchain-ai.github.io/langgraph/how-tos/state-model/) as your graph state to add **default values** and additional data validation.

By default, the graph will have the same input and output schemas. If you want to change this, you can also specify explicit input and output schemas directly. This is useful when you have a lot of keys, and some are explicitly for input and others for output. See the [notebook here](https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/) for how to use.

#### Multiple schemas[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas "Permanent link")

Typically, all graph nodes communicate with a single schema. This means that they will read and write to the same state channels. But, there are cases where we want more control over this:

- Internal nodes can pass information that is not required in the graph's input / output.
- We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

It is possible to have nodes write to private state channels inside the graph for internal node communication. We can simply define a private schema, `PrivateState`. See [this notebook](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/) for more detail.

It is also possible to define explicit input and output schemas for a graph. In these cases, we define an "internal" schema that contains _all_ keys relevant to graph operations. But, we also define `input` and `output` schemas that are sub-sets of the "internal" schema to constrain the input and output of the graph. See [this notebook](https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/) for more detail.

Let's look at an example:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-1)class InputState(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-2)    user_input: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-3)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-4)class OutputState(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-5)    graph_output: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-6)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-7)class OverallState(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-8)    foo: str    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-9)    user_input: str    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-10)    graph_output: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-11)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-12)class PrivateState(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-13)    bar: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-14)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-15)def node_1(state: InputState) -> OverallState:     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-16)    # Write to OverallState    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-17)    return {"foo": state["user_input"] + " name"} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-18)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-19)def node_2(state: OverallState) -> PrivateState:     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-20)    # Read from OverallState, write to PrivateState    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-21)    return {"bar": state["foo"] + " is"} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-22)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-23)def node_3(state: PrivateState) -> OutputState:     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-24)    # Read from PrivateState, write to OutputState    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-25)    return {"graph_output": state["bar"] + " Lance"} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-26)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-27)builder = StateGraph(OverallState,input=InputState,output=OutputState) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-28)builder.add_node("node_1", node_1) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-29)builder.add_node("node_2", node_2) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-30)builder.add_node("node_3", node_3) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-31)builder.add_edge(START, "node_1") [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-32)builder.add_edge("node_1", "node_2") [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-33)builder.add_edge("node_2", "node_3") [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-34)builder.add_edge("node_3", END) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-35)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-36)graph = builder.compile() [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-37)graph.invoke({"user_input":"My"}) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-1-38){'graph_output': 'My name is Lance'}`

There are two subtle and important points to note here:

1.  We pass `state: InputState` as the input schema to `node_1`. But, we write out to `foo`, a channel in `OverallState`. How can we write out to a state channel that is not included in the input schema? This is because a node _can write to any state channel in the graph state._ The graph state is the union of of the state channels defined at initialization, which includes `OverallState` and the filters `InputState` and `OutputState`.
2.  We initialize the graph with `StateGraph(OverallState,input=InputState,output=OutputState)`. So, how can we write to `PrivateState` in `node_2`? How does the graph gain access to this schema if it was not passed in the `StateGraph` initialization? We can do this because _nodes can also declare additional state channels_ as long as the state schema definition exists. In this case, the `PrivateState` schema is defined, so we can add `bar` as a new state channel in the graph and write to it.

### Reducers[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers "Permanent link")

Reducers are key to understanding how updates from nodes are applied to the `State`. Each key in the `State` has its own independent reducer function. If no reducer function is explicitly specified then it is assumed that all updates to that key should override it. There are a few different types of reducers, starting with the default type of reducer:

#### Default Reducer[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#default-reducer "Permanent link")

These two examples show how to use the default reducer:

**Example A:**

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-2-1)from typing_extensions import TypedDict [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-2-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-2-3)class State(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-2-4)    foo: int    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-2-5)    bar: list[str]`

In this example, no reducer functions are specified for any key. Let's assume the input to the graph is `{"foo": 1, "bar": ["hi"]}`. Let's then assume the first `Node` returns `{"foo": 2}`. This is treated as an update to the state. Notice that the `Node` does not need to return the whole `State` schema - just an update. After applying this update, the `State` would then be `{"foo": 2, "bar": ["hi"]}`. If the second node returns `{"bar": ["bye"]}` then the `State` would then be `{"foo": 2, "bar": ["bye"]}`

**Example B:**

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-3-1)from typing import Annotated [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-3-2)from typing_extensions import TypedDict [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-3-3)from operator import add [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-3-4)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-3-5)class State(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-3-6)    foo: int    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-3-7)    bar: Annotated[list[str], add]`

In this example, we've used the `Annotated` type to specify a reducer function (`operator.add`) for the second key (`bar`). Note that the first key remains unchanged. Let's assume the input to the graph is `{"foo": 1, "bar": ["hi"]}`. Let's then assume the first `Node` returns `{"foo": 2}`. This is treated as an update to the state. Notice that the `Node` does not need to return the whole `State` schema - just an update. After applying this update, the `State` would then be `{"foo": 2, "bar": ["hi"]}`. If the second node returns `{"bar": ["bye"]}` then the `State` would then be `{"foo": 2, "bar": ["hi", "bye"]}`. Notice here that the `bar` key is updated by adding the two lists together.

### Working with Messages in Graph State[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state "Permanent link")

#### Why use messages?[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#why-use-messages "Permanent link")

Most modern LLM providers have a chat model interface that accepts a list of messages as input. LangChain's [`ChatModel`](https://python.langchain.com/docs/concepts/#chat-models) in particular accepts a list of `Message` objects as inputs. These messages come in a variety of forms such as `HumanMessage` (user input) or `AIMessage` (LLM response). To read more about what message objects are, please refer to [this](https://python.langchain.com/docs/concepts/#messages) conceptual guide.

#### Using Messages in your Graph[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#using-messages-in-your-graph "Permanent link")

In many cases, it is helpful to store prior conversation history as a list of messages in your graph state. To do so, we can add a key (channel) to the graph state that stores a list of `Message` objects and annotate it with a reducer function (see `messages` key in the example below). The reducer function is vital to telling the graph how to update the list of `Message` objects in the state with each state update (for example, when a node sends an update). If you don't specify a reducer, every state update will overwrite the list of messages with the most recently provided value. If you wanted to simply append messages to the existing list, you could use `operator.add` as a reducer.

However, you might also want to manually update messages in your graph state (e.g. human-in-the-loop). If you were to use `operator.add`, the manual state updates you send to the graph would be appended to the existing list of messages, instead of updating existing messages. To avoid that, you need a reducer that can keep track of message IDs and overwrite existing messages, if updated. To achieve this, you can use the prebuilt `add_messages` function. For brand new messages, it will simply append to existing list, but it will also handle the updates for existing messages correctly.

#### Serialization[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#serialization "Permanent link")

In addition to keeping track of message IDs, the `add_messages` function will also try to deserialize messages into LangChain `Message` objects whenever a state update is received on the `messages` channel. See more information on LangChain serialization/deserialization [here](https://python.langchain.com/docs/how_to/serialization/). This allows sending graph inputs / state updates in the following format:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-4-1)# this is supported [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-4-2){"messages": [HumanMessage(content="message")]} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-4-3)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-4-4)# and this is also supported [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-4-5){"messages": [{"type": "human", "content": "message"}]}`

Since the state updates are always deserialized into LangChain `Messages` when using `add_messages`, you should use dot notation to access message attributes, like `state["messages"][-1].content`. Below is an example of a graph that uses `add_messages` as it's reducer function.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-5-1)from langchain_core.messages import AnyMessage [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-5-2)from langgraph.graph.message import add_messages [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-5-3)from typing import Annotated [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-5-4)from typing_extensions import TypedDict [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-5-5)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-5-6)class GraphState(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-5-7)    messages: Annotated[list[AnyMessage], add_messages]`

API Reference: [AnyMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.AnyMessage.html) | [add_messages](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages)

#### MessagesState[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate "Permanent link")

Since having a list of messages in your state is so common, there exists a prebuilt state called `MessagesState` which makes it easy to use messages. `MessagesState` is defined with a single `messages` key which is a list of `AnyMessage` objects and uses the `add_messages` reducer. Typically, there is more state to track than just messages, so we see people subclass this state and add more fields, like:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-6-1)from langgraph.graph import MessagesState [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-6-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-6-3)class State(MessagesState):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-6-4)    documents: list[str]`

## Nodes[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes "Permanent link")

In LangGraph, nodes are typically python functions (sync or async) where the **first** positional argument is the [state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state), and (optionally), the **second** positional argument is a "config", containing optional [configurable parameters](https://langchain-ai.github.io/langgraph/concepts/low_level/#configuration) (such as a `thread_id`).

Similar to `NetworkX`, you add these nodes to a graph using the [add_node](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_node) method:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-1)from langchain_core.runnables import RunnableConfig [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-2)from langgraph.graph import StateGraph [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-3)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-4)builder = StateGraph(dict) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-5) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-6)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-7)def my_node(state: dict, config: RunnableConfig):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-8)    print("In node: ", config["configurable"]["user_id"])    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-9)    return {"results": f"Hello, {state['input']}!"} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-10) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-11)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-12)# The second argument is optional [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-13)def my_other_node(state: dict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-14)    return state [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-15) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-16)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-17)builder.add_node("my_node", my_node) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-18)builder.add_node("other_node", my_other_node) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-7-19)...`

API Reference: [RunnableConfig](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html) | [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)

Behind the scenes, functions are converted to [RunnableLambda](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableLambda.html#langchain_core.runnables.base.RunnableLambda)s, which add batch and async support to your function, along with native tracing and debugging.

If you add a node to a graph without specifying a name, it will be given a default name equivalent to the function name.

`` [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-8-1)builder.add_node(my_node) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-8-2)# You can then create edges to/from this node by referencing it as `"my_node"` ``

### `START` Node[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#start-node "Permanent link")

The `START` Node is a special node that represents the node that sends user input to the graph. The main purpose for referencing this node is to determine which nodes should be called first.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-9-1)from langgraph.graph import START [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-9-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-9-3)graph.add_edge(START, "node_a")`

API Reference: [START](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START)

### `END` Node[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#end-node "Permanent link")

The `END` Node is a special node that represents a terminal node. This node is referenced when you want to denote which edges have no actions after they are done.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-10-1)from langgraph.graph import END [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-10-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-10-3)graph.add_edge("node_a", END)`

## Edges[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges "Permanent link")

Edges define how the logic is routed and how the graph decides to stop. This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:

- Normal Edges: Go directly from one node to the next.
- Conditional Edges: Call a function to determine which node(s) to go to next.
- Entry Point: Which node to call first when user input arrives.
- Conditional Entry Point: Call a function to determine which node(s) to call first when user input arrives.

A node can have MULTIPLE outgoing edges. If a node has multiple out-going edges, **all** of those destination nodes will be executed in parallel as a part of the next superstep.

### Normal Edges[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#normal-edges "Permanent link")

If you **always** want to go from node A to node B, you can use the [add_edge](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_edge) method directly.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-11-1)graph.add_edge("node_a", "node_b")`

### Conditional Edges[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges "Permanent link")

If you want to **optionally** route to 1 or more edges (or optionally terminate), you can use the [add_conditional_edges](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.Graph.add_conditional_edges) method. This method accepts the name of a node and a "routing function" to call after that node is executed:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-12-1)graph.add_conditional_edges("node_a", routing_function)`

Similar to nodes, the `routing_function` accepts the current `state` of the graph and returns a value.

By default, the return value `routing_function` is used as the name of the node (or list of nodes) to send the state to next. All those nodes will be run in parallel as a part of the next superstep.

You can optionally provide a dictionary that maps the `routing_function`'s output to the name of the next node.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-13-1)graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})`

Tip

Use [`Command`](https://langchain-ai.github.io/langgraph/concepts/low_level/#command) instead of conditional edges if you want to combine state updates and routing in a single function.

### Entry Point[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#entry-point "Permanent link")

The entry point is the first node(s) that are run when the graph starts. You can use the [`add_edge`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_edge) method from the virtual [`START`](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START) node to the first node to execute to specify where to enter the graph.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-14-1)from langgraph.graph import START [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-14-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-14-3)graph.add_edge(START, "node_a")`

API Reference: [START](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START)

### Conditional Entry Point[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-entry-point "Permanent link")

A conditional entry point lets you start at different nodes depending on custom logic. You can use [`add_conditional_edges`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.Graph.add_conditional_edges) from the virtual [`START`](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START) node to accomplish this.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-15-1)from langgraph.graph import START [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-15-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-15-3)graph.add_conditional_edges(START, routing_function)`

API Reference: [START](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START)

You can optionally provide a dictionary that maps the `routing_function`'s output to the name of the next node.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-16-1)graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})`

## `Send`[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#send "Permanent link")

By default, `Nodes` and `Edges` are defined ahead of time and operate on the same shared state. However, there can be cases where the exact edges are not known ahead of time and/or you may want different versions of `State` to exist at the same time. A common example of this is with `map-reduce` design patterns. In this design pattern, a first node may generate a list of objects, and you may want to apply some other node to all those objects. The number of objects may be unknown ahead of time (meaning the number of edges may not be known) and the input `State` to the downstream `Node` should be different (one for each generated object).

To support this design pattern, LangGraph supports returning [`Send`](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Send) objects from conditional edges. `Send` takes two arguments: first is the name of the node, and second is the state to pass to that node.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-17-1)def continue_to_jokes(state: OverallState):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-17-2)    return [Send("generate_joke", {"subject": s}) for s in state['subjects']] [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-17-3)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-17-4)graph.add_conditional_edges("node_a", continue_to_jokes)`

## `Command`[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#command "Permanent link")

It can be useful to combine control flow (edges) and state updates (nodes). For example, you might want to BOTH perform state updates AND decide which node to go to next in the SAME node. LangGraph provides a way to do so by returning a [`Command`](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command) object from node functions:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-18-1)def my_node(state: State) -> Command[Literal["my_other_node"]]:     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-18-2)    return Command(        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-18-3)        # state update        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-18-4)        update={"foo": "bar"},        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-18-5)        # control flow        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-18-6)        goto="my_other_node"    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-18-7)    )`

With `Command` you can also achieve dynamic control flow behavior (identical to [conditional edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)):

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-19-1)def my_node(state: State) -> Command[Literal["my_other_node"]]:     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-19-2)    if state["foo"] == "bar":        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-19-3)        return Command(update={"foo": "baz"}, goto="my_other_node")`

Important

When returning `Command` in your node functions, you must add return type annotations with the list of node names the node is routing to, e.g. `Command[Literal["my_other_node"]]`. This is necessary for the graph rendering and tells LangGraph that `my_node` can navigate to `my_other_node`.

Check out this [how-to guide](https://langchain-ai.github.io/langgraph/how-tos/command/) for an end-to-end example of how to use `Command`.

### When should I use Command instead of conditional edges?[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#when-should-i-use-command-instead-of-conditional-edges "Permanent link")

Use `Command` when you need to **both** update the graph state **and** route to a different node. For example, when implementing [multi-agent handoffs](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#handoffs) where it's important to route to a different agent and pass some information to that agent.

Use [conditional edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges) to route between nodes conditionally without updating the state.

### Navigating to a node in a parent graph[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#navigating-to-a-node-in-a-parent-graph "Permanent link")

If you are using [subgraphs](https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs), you might want to navigate from a node within a subgraph to a different subgraph (i.e. a different node in the parent graph). To do so, you can specify `graph=Command.PARENT` in `Command`:

`` [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-20-1)def my_node(state: State) -> Command[Literal["my_other_node"]]:     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-20-2)    return Command(        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-20-3)        update={"foo": "bar"},        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-20-4)        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-20-5)        graph=Command.PARENT    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-20-6)    ) ``

Note

Setting `graph` to `Command.PARENT` will navigate to the closest parent graph.

State updates with `Command.PARENT`

When you send updates from a subgraph node to a parent graph node for a key that's shared by both parent and subgraph [state schemas](https://langchain-ai.github.io/langgraph/concepts/low_level/#schema), you **must** define a [reducer](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) for the key you're updating in the parent graph state. See this [example](https://langchain-ai.github.io/langgraph/how-tos/command/#navigating-to-a-node-in-a-parent-graph).

This is particularly useful when implementing [multi-agent handoffs](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#handoffs).

### Using inside tools[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#using-inside-tools "Permanent link")

A common use case is updating graph state from inside a tool. For example, in a customer support application you might want to look up customer information based on their account number or ID in the beginning of the conversation. To update the graph state from the tool, you can return `Command(update={"my_custom_key": "foo", "messages": [...]})` from the tool:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-1)@tool [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-2)def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-3)    """Use this to look up user information to better assist them with their questions."""    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-4)    user_info = get_user_info(config.get("configurable", {}).get("user_id"))    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-5)    return Command(        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-6)        update={            [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-7)            # update the state keys            [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-8)            "user_info": user_info,            [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-9)            # update the message history            [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-10)            "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-11)        }    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-21-12)    )`

Important

You MUST include `messages` (or any state key used for the message history) in `Command.update` when returning `Command` from a tool and the list of messages in `messages` MUST contain a `ToolMessage`. This is necessary for the resulting message history to be valid (LLM providers require AI messages with tool calls to be followed by the tool result messages).

If you are using tools that update state via `Command`, we recommend using prebuilt [`ToolNode`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode) which automatically handles tools returning `Command` objects and propagates them to the graph state. If you're writing a custom node that calls tools, you would need to manually propagate `Command` objects returned by the tools as the update from the node.

### Human-in-the-loop[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#human-in-the-loop "Permanent link")

`Command` is an important part of human-in-the-loop workflows: when using `interrupt()` to collect user input, `Command` is then used to supply the input and resume execution via `Command(resume="User input")`. Check out [this conceptual guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) for more information.

## Persistence[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#persistence "Permanent link")

LangGraph provides built-in persistence for your agent's state using [checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver). Checkpointers save snapshots of the graph state at every superstep, allowing resumption at any time. This enables features like human-in-the-loop interactions, memory management, and fault-tolerance. You can even directly manipulate a graph's state after its execution using the appropriate `get` and `update` methods. For more details, see the [persistence conceptual guide](https://langchain-ai.github.io/langgraph/concepts/persistence/).

## Threads[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#threads "Permanent link")

Threads in LangGraph represent individual sessions or conversations between your graph and a user. When using checkpointing, turns in a single conversation (and even steps within a single graph execution) are organized by a unique thread ID.

## Storage[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#storage "Permanent link")

LangGraph provides built-in document storage through the [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) interface. Unlike checkpointers, which save state by thread ID, stores use custom namespaces for organizing data. This enables cross-thread persistence, allowing agents to maintain long-term memories, learn from past interactions, and accumulate knowledge over time. Common use cases include storing user profiles, building knowledge bases, and managing global preferences across all threads.

## Graph Migrations[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#graph-migrations "Permanent link")

LangGraph can easily handle migrations of graph definitions (nodes, edges, and state) even when using a checkpointer to track state.

- For threads at the end of the graph (i.e. not interrupted) you can change the entire topology of the graph (i.e. all nodes and edges, remove, add, rename, etc)
- For threads currently interrupted, we support all topology changes other than renaming / removing nodes (as that thread could now be about to enter a node that no longer exists) -- if this is a blocker please reach out and we can prioritize a solution.
- For modifying state, we have full backwards and forwards compatibility for adding and removing keys
- State keys that are renamed lose their saved state in existing threads
- State keys whose types change in incompatible ways could currently cause issues in threads with state from before the change -- if this is a blocker please reach out and we can prioritize a solution.

## Configuration[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#configuration "Permanent link")

When creating a graph, you can also mark that certain parts of the graph are configurable. This is commonly done to enable easily switching between models or system prompts. This allows you to create a single "cognitive architecture" (the graph) but have multiple different instance of it.

You can optionally specify a `config_schema` when creating a graph.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-22-1)class ConfigSchema(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-22-2)    llm: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-22-3)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-22-4)graph = StateGraph(State, config_schema=ConfigSchema)`

You can then pass this configuration into the graph using the `configurable` config field.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-23-1)config = {"configurable": {"llm": "anthropic"}} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-23-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-23-3)graph.invoke(inputs, config=config)`

You can then access and use this configuration inside a node:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-24-1)def node_a(state, config):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-24-2)    llm_type = config.get("configurable", {}).get("llm", "openai")    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-24-3)    llm = get_llm(llm_type)    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-24-4)    ...`

See [this guide](https://langchain-ai.github.io/langgraph/how-tos/configuration/) for a full breakdown on configuration.

### Recursion Limit[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#recursion-limit "Permanent link")

The recursion limit sets the maximum number of [super-steps](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs) the graph can execute during a single execution. Once the limit is reached, LangGraph will raise `GraphRecursionError`. By default this value is set to 25 steps. The recursion limit can be set on any graph at runtime, and is passed to `.invoke`/`.stream` via the config dictionary. Importantly, `recursion_limit` is a standalone `config` key and should not be passed inside the `configurable` key as all other user-defined configuration. See the example below:

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-25-1)graph.invoke(inputs, config={"recursion_limit": 5, "configurable":{"llm": "anthropic"}})`

Read [this how-to](https://langchain-ai.github.io/langgraph/how-tos/recursion-limit/) to learn more about how the recursion limit works.

## `interrupt`[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#interrupt "Permanent link")

Use the [interrupt](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt) function to **pause** the graph at specific points to collect user input. The `interrupt` function surfaces interrupt information to the client, allowing the developer to collect user input, validate the graph state, or make decisions before resuming execution.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-1)from langgraph.types import interrupt [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-3)def human_approval_node(state: State):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-4)    ...    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-5)    answer = interrupt(        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-6)        # This value will be sent to the client.        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-7)        # It can be any JSON serializable value.        [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-8)        {"question": "is it ok to continue?"},    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-9)    )    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-26-10)    ...`

API Reference: [interrupt](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt)

Resuming the graph is done by passing a [`Command`](https://langchain-ai.github.io/langgraph/concepts/low_level/#command) object to the graph with the `resume` key set to the value returned by the `interrupt` function.

Read more about how the `interrupt` is used for **human-in-the-loop** workflows in the [Human-in-the-loop conceptual guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).

## Breakpoints[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#breakpoints "Permanent link")

Breakpoints pause graph execution at specific points and enable stepping through execution step by step. Breakpoints are powered by LangGraph's [**persistence layer**](https://langchain-ai.github.io/langgraph/concepts/persistence/), which saves the state after each graph step. Breakpoints can also be used to enable [**human-in-the-loop**](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) workflows, though we recommend using the [`interrupt` function](https://langchain-ai.github.io/langgraph/concepts/low_level/#interrupt) for this purpose.

Read more about breakpoints in the [Breakpoints conceptual guide](https://langchain-ai.github.io/langgraph/concepts/breakpoints/).

## Subgraphs[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs "Permanent link")

A subgraph is a [graph](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs) that is used as a [node](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes) in another graph. This is nothing more than the age-old concept of encapsulation, applied to LangGraph. Some reasons for using subgraphs are:

- building [multi-agent systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- when you want to reuse a set of nodes in multiple graphs, which maybe share some state, you can define them once in a subgraph and then use them in multiple parent graphs
- when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

There are two ways to add subgraphs to a parent graph:

- add a node with the compiled subgraph: this is useful when the parent graph and the subgraph share state keys and you don't need to transform state on the way in or out

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-27-1)builder.add_node("subgraph", subgraph_builder.compile())`

- add a node with a function that invokes the subgraph: this is useful when the parent graph and the subgraph have different state schemas and you need to transform state before or after calling the subgraph

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-28-1)subgraph = subgraph_builder.compile() [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-28-2)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-28-3)def call_subgraph(state: State):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-28-4)    return subgraph.invoke({"subgraph_key": state["parent_key"]}) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-28-5)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-28-6)builder.add_node("subgraph", call_subgraph)`

Let's take a look at examples for each.

### As a compiled graph[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#as-a-compiled-graph "Permanent link")

The simplest way to create subgraph nodes is by using a [compiled subgraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#compiling-your-graph) directly. When doing so, it is **important** that the parent graph and the subgraph [state schemas](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) share at least one key which they can use to communicate. If your graph and subgraph do not share any keys, you should write a function [invoking the subgraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#as-a-function) instead.

Note

If you pass extra keys to the subgraph node (i.e., in addition to the shared keys), they will be ignored by the subgraph node. Similarly, if you return extra keys from the subgraph, they will be ignored by the parent graph.

`[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-1)from langgraph.graph import StateGraph [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-2)from typing import TypedDict [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-3)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-4)class State(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-5)    foo: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-6)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-7)class SubgraphState(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-8)    foo: str  # note that this key is shared with the parent graph state    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-9)    bar: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-10)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-11)# Define subgraph [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-12)def subgraph_node(state: SubgraphState):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-13)    # note that this subgraph node can communicate with the parent graph via the shared "foo" key    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-14)    return {"foo": state["foo"] + "bar"} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-15)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-16)subgraph_builder = StateGraph(SubgraphState) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-17)subgraph_builder.add_node(subgraph_node) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-18)... [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-19)subgraph = subgraph_builder.compile() [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-20)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-21)# Define parent graph [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-22)builder = StateGraph(State) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-23)builder.add_node("subgraph", subgraph) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-24)... [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-29-25)graph = builder.compile()`

API Reference: [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)

### As a function[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#as-a-function "Permanent link")

You might want to define a subgraph with a completely different schema. In this case, you can create a node function that invokes the subgraph. This function will need to [transform](https://langchain-ai.github.io/langgraph/how-tos/subgraph-transform-state/) the input (parent) state to the subgraph state before invoking the subgraph, and transform the results back to the parent state before returning the state update from the node.

`` [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-1)class State(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-2)    foo: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-3)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-4)class SubgraphState(TypedDict):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-5)    # note that none of these keys are shared with the parent graph state    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-6)    bar: str    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-7)    baz: str [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-8)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-9)# Define subgraph [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-10)def subgraph_node(state: SubgraphState):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-11)    return {"bar": state["bar"] + "baz"} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-12)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-13)subgraph_builder = StateGraph(SubgraphState) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-14)subgraph_builder.add_node(subgraph_node) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-15)... [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-16)subgraph = subgraph_builder.compile() [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-17)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-18)# Define parent graph [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-19)def node(state: State):     [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-20)    # transform the state to the subgraph state    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-21)    response = subgraph.invoke({"bar": state["foo"]})    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-22)    # transform response back to the parent state    [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-23)    return {"foo": response["bar"]} [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-24)[](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-25)builder = StateGraph(State) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-26)# note that we are using `node` function instead of a compiled subgraph [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-27)builder.add_node(node) [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-28)... [](https://langchain-ai.github.io/langgraph/concepts/low_level/#__codelineno-30-29)graph = builder.compile() ``

## Visualization[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#visualization "Permanent link")

It's often nice to be able to visualize graphs, especially as they get more complex. LangGraph comes with several built-in ways to visualize graphs. See [this how-to guide](https://langchain-ai.github.io/langgraph/how-tos/visualization/) for more info.

## Streaming[¶](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming "Permanent link")

LangGraph is built with first class support for streaming, including streaming updates from graph nodes during the execution, streaming tokens from LLM calls and more. See this [conceptual guide](https://langchain-ai.github.io/langgraph/concepts/streaming/) for more information.
