import torch
from dotenv import load_dotenv
from langchain.agents import create_agent  # ty: ignore
from langchain_chroma import Chroma  # ty: ignore
from langchain_core.tools import tool  # ty: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # ty: ignore

load_dotenv()

# 加载嵌入模型
embedding = HuggingFaceEmbeddings(
    model_name="./bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 向量库
vector_store = Chroma(
    collection_name="example_rag",
    embedding_function=embedding,
    persist_directory="./chroma_rag_db",  # 目录
)

system_prompt = """
    你可以使用信息检索工具，回答用户的问题。
"""


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve infomation to help answer a query"""
    retrieve_docs = vector_store.similarity_search(query, k=2)
    # 提取文档内容作为消息内容
    content = "\n\n".join([doc.page_content for doc in retrieve_docs])
    # 返回元组：(处理后的内容, 原始文档)
    return content, retrieve_docs


agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[retrieve_context],
    system_prompt=system_prompt,
)


results = agent.invoke({"messages": [{"role": "user", "content": "讲一下3i/Atlas"}]})

messages = results["messages"]
print(f"历史消息：{len(messages)}条")
for message in messages:
    message.pretty_print()
