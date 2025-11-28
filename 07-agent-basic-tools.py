from dotenv import load_dotenv
from langchain.agents import create_agent  # ty:ignore

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It is always sunny in {city}"


agent = create_agent(
    model="deepseek:deepseek-chat",  # 不能写provider
    tools=[get_weather],
)

results = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}
)
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()  # 漂亮输出
