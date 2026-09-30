
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.state import RAGState

def split_node(state: RAGState):

    documents = state["documents"]  #split_node 不自己去读 PDF、Excel、PPT。它只从 State 中拿 load_node 已经准备好的 documents

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("文件中没有可索引的文本内容")

    return {
        "chunks": chunks
    }
