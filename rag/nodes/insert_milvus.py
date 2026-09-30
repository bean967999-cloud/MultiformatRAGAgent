import os
import logging

from dotenv import load_dotenv

from rag.config import collection_name, index_batch_size
from rag.embeddings.factory import EmbeddingFactory
from rag.state import RAGState
from rag.vectorstores.milvus_client import MilvusVectorStore

load_dotenv()
logger = logging.getLogger(__name__)

def insert_node(state: RAGState):
    """Embed and upsert chunks in bounded batches."""
    chunks = state["chunks"]
    uri = os.getenv("MILVUS_URI")
    if not uri:
        raise ValueError("请在 .env 中配置 MILVUS_URI")
    batch_size = index_batch_size()
    embedding = EmbeddingFactory.create()
    store = MilvusVectorStore(
        uri=uri,
        token=os.getenv("MILVUS_TOKEN")
    )
    name = collection_name()
    logger.info("开始入库：collection=%s chunks=%d batch_size=%d", name, len(chunks), batch_size)
    result = store.replace_document_stream(
        chunks=chunks,
        embed_documents=embedding.embed_documents,
        source=str(state["file_path"]),
        collection_name=name,
        batch_size=batch_size,
    )
    logger.info("入库完成：upsert=%d obsolete=%d", result["upsert_count"], result["deleted_count"])
    return {"inserted_count": result["upsert_count"]}
