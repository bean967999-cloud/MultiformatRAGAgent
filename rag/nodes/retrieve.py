import os

from dotenv import load_dotenv

from rag.config import collection_name
from rag.embeddings.factory import EmbeddingFactory
from rag.state import RAGState
from rag.vectorstores.milvus_client import MilvusVectorStore

load_dotenv()


def retrieve_node(state: RAGState):
    question = state["question"].strip()
    if not question:
        raise ValueError("问题不能为空")
    embedding = EmbeddingFactory.create()
    query_vector = embedding.embed_query(question)
    uri = os.getenv("MILVUS_URI")
    if not uri:
        raise ValueError("请在 .env 中配置 MILVUS_URI")
    store = MilvusVectorStore(uri=uri, token=os.getenv("MILVUS_TOKEN"))
    return {"retrieved_documents": store.search_documents(query_vector, collection_name=collection_name())}
