from pathlib import Path
import logging

from rag.state import RAGState
from rag.services.document_loader_service import DocumentLoaderService

logger = logging.getLogger(__name__)

def load_node(state: RAGState):

    file_path = Path(state["file_path"]).expanduser().resolve(strict=True)

    #调用文档加载服务
    documents = DocumentLoaderService.load(file_path)

    logger.info("已加载文件 %s：%d 个 Document", file_path, len(documents))

    return {
        "documents": documents,
        "file_path": file_path,
    }
