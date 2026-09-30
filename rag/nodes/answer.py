from rag.llm import answer_with_context
from rag.state import RAGState


def answer_node(state: RAGState):
    documents = state["retrieved_documents"]
    if not documents:
        return {"answer": "未检索到相关资料，无法根据知识库回答。"}
    context = "\n\n".join(
        f"[{index}] 来源: {doc.metadata.get('source', '未知')} "
        f"页码/位置: {doc.metadata.get('page', doc.metadata.get('slide', doc.metadata.get('row', '未知')))}\n"
        f"{doc.page_content}"
        for index, doc in enumerate(documents, 1)
    )
    return {"answer": answer_with_context(state["question"], context)}
