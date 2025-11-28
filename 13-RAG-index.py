import bs4  # 读取网页
import torch
from langchain_chroma import Chroma  # ty: ignore
from langchain_community.document_loaders import WebBaseLoader  # ty: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # ty: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ty: ignore

# 1.读取网页
page_url = "https://news.cctv.cn/2025/08/07/ARTIwHXTjBUTWQHIhY3pmv7Z250807.shtml"
bs4_strainer = bs4.SoupStrainer()
loader = WebBaseLoader(web_path=(page_url), bs_kwargs={"parse_only": bs4_strainer})

docs = loader.load()

# 2.文本切块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 1000个字符为1块
    chunk_overlap=200,
    add_start_index=True,  # 给块添加索引
)
all_splits = text_splitter.split_documents(docs)  # list[Document]

# 3.加载嵌入模型
embedding = HuggingFaceEmbeddings(
    model_name="./bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)

# 4.向量和文本块的存储
vector_store = Chroma(
    collection_name="example_rag",
    embedding_function=embedding,
    persist_directory="./chroma_rag_db",  # 目录
)
ids = vector_store.add_documents(all_splits)  # 文本加向量
print(len(ids))
print(ids)
