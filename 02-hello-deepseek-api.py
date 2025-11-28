# # 旧方法
# from dotenv import load_dotenv
# from langchain_deepseek import ChatDeepSeek

# load_dotenv()
# model = ChatDeepSeek(
#     model="deepseek-chat",  # deepseek-reasoner
#     temperature=0.1,
#     max_tokens=100,
#     timeout=None,
#     max_retries=3,
# )
# for chuck in model.stream("来一段毛泽东的诗词"):
#     print(chuck.content, end="", flush=True)

# 新方法
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model  # ty: ignore

load_dotenv()
# 初始化聊天模型
model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    temperature=0.1,
    max_tokens=1000,
    timeout=None,
    max_retries=3,
)
for chuck in model.stream("来一段毛主席诗词"):
    print(chuck.content, end="", flush=True)
