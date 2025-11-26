import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1.读取文本
pdf_documents = PyPDFLoader("分数阶反卷积的高分辨力目标亮点提取.pdf").load()
# print(len(pdf_documents))  # 打印多少页
# print(pdf_documents[0])  # 打印第一页内容

# 2.文本切块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 1000个字符为1块
    chunk_overlap=200,
    add_start_index=True,  # 给块添加索引
)
all_splits = text_splitter.split_documents(pdf_documents)  # list[Document]
# print(len(all_splits))
# print(all_splits[0])

# 3.加载嵌入模型
embedding = HuggingFaceEmbeddings(
    model_name="./bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)
# vector_0 = embedding.embed_query(all_splits[0].page_content)
# print(len(vector_0))
# print(vector_0)

# 4.向量和文本块的存储
vector_store = Chroma(
    collection_name="example",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db",  # 目录
)
ids = vector_store.add_documents(all_splits)  # 文本加向量
print(len(ids))
print(ids)
