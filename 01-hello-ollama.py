from langchain.chat_models import init_chat_model  # ty: ignore
from langchain_ollama import ChatOllama  # ty: ignore

# 旧方法
model = ChatOllama(
    model="gemma3:4b", base_url="http://localhost:11434", temperature=0.1
)
for chuck in model.stream("来一段唐诗"):
    print(chuck.content, end="", flush=True)

print("\n" + "=" * 50 + "\n")

# 新方法(推荐)

model = init_chat_model(
    model="gemma3:4b",
    model_provider="ollama",
    base_url="http://localhost:11434",
    temperature=0.1,
    timeout=30,
    max_tokens=2200,
)
for chuck in model.stream("来一段宋词"):
    print(chuck.content, end="", flush=True)
