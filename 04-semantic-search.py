from typing import List

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_huggingface import HuggingFaceEmbeddings

# 1.构建嵌入模型
embedding = HuggingFaceEmbeddings(
    model_name="./bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)
# 2.向量库
vector_store = Chroma(
    embedding_function=embedding,
    collection_name="example",
    persist_directory="./chroma_langchain_db",
)
# 3.1相似度查询
# results = vector_store.similarity_search(query="仿真实验怎么设置的？", k=3)
# print(results)
# 3.2带分数的相似度查询（欧氏距离，越小越好）
# results = vector_store.similarity_search_with_score(query="仿真实验怎么设置的？", k=3)
# print(results)
# 3.3用向量进行相似度查询(等价于第一种)
# vector = embedding.embed_query("仿真实验怎么设置的？")
# results = vector_store.similarity_search_by_vector(embedding=vector, k=3)
# print(results)


# 3.4封装chain
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


results = retriever.invoke("仿真实验怎么设置的？")
print(results)
