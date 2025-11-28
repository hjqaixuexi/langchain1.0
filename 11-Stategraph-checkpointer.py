from operator import add
from typing import Annotated

from langchain_core.runnables import RunnableConfig  # ty: ignore
from langgraph.checkpoint.memory import InMemorySaver  # ty: ignore
from langgraph.graph import END, START, StateGraph  # ty: ignore
from typing_extensions import TypedDict


# 表达状态
class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


# 构建状态图
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 检查点管理器
checkpointer = InMemorySaver()

# 编译
graph = workflow.compile(checkpointer=checkpointer)

# 配置
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# 调用
results = graph.invoke({"foo": ""}, config)
print(results)  # {'foo': 'b', 'bar': ['a', 'b']}

# 状态查看
print(graph.get_state(config))
