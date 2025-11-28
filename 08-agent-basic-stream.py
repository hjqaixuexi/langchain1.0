from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It is always sunny in {city}"


agent = create_agent(
    model="deepseek:deepseek-chat",  # 不能写provider
    tools=[get_weather],
)

# 第一种
# for event in agent.stream(
#    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
#     stream_mode="values",  # 以消息为单位输出
# ):
#     messages = event["messages"]
#     print(f"历史消息：{len(messages)}条")
#     messages[-1].pretty_print()  # 漂亮输出

# 第二种
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",  # 以token为单位输出
):
    print(chunk[0].content, end=" ")
