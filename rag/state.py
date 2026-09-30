from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document

#TypedDict规定一个字典应该有哪些 Key，以及每个 Key 对应什么类型
#TypedDict更重要的作用是：告诉开发者和类型检查工具，这个字典应该长什么样
class RAGState(TypedDict,total=False):
    file_path:Path
    documents: list[Document]
    chunks: list[Document]
    dense_vectors: list[list[float]]
    inserted_count: int
    question: str
    retrieved_documents: list[Document]
    answer: str
