from dotenv import load_dotenv
from langchain.agents import create_agent  # ty: ignore

load_dotenv()
agent = create_agent(
    model="deepseek:deepseek-chat",
)

# 历史消息列表
history_message = []

# 第一轮
results = agent.invoke({"messages": [{"role": "user", "content": "来一首宋词"}]})
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()

history_message = messages

# 第二轮
message = {"role": "user", "content": "再来"}
history_message.append(message)
results = agent.invoke({"messages": history_message})
messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()
