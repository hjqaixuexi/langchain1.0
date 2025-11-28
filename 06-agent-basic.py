from dotenv import load_dotenv
from langchain.agents import create_agent  # ty: ignore

load_dotenv()
agent = create_agent(
    model="deepseek:deepseek-chat",  # 不能写provider
)

# print(agent)  # langgraph.graph.state.CompiledStateGraph,图结构
# print(agent.nodes)
results = agent.invoke(
    {"messages": [{"role": "user", "content": "明天旧金山天气如何"}]}
)
# print(results)  # AIMessage/HumanMessage
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()  # 漂亮输出
