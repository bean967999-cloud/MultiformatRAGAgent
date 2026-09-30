'''
Embedding 模型通过神经网络把文本映射到一个高维向量空间，使语义相近的文本在向量空间中的距离更近，从而支持语义检索。

'''

from rag.state import RAGState
from rag.embeddings.factory import EmbeddingFactory

def embedding_node(state: RAGState):

    #将切分后的chunks转换成Embedding向量
    print("------------embedding node------------")

    chunks = state["chunks"]

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embedding = EmbeddingFactory.create()

    #create() 负责创建 Embedding 对象；embed_documents() 是这个 Embedding 对象提供的“批量文本转向量”能力。
    dense_vectors = embedding.embed_documents(texts)

    print("chunk 数量：", len(chunks))
    print("向量数量：", len(dense_vectors))

    if dense_vectors:
        print("向量维度：", len(dense_vectors[0]))

    #Node 应该负责“返回本次状态更新”，由 LangGraph 把它合并进整个 state
    return {
        "dense_vectors": dense_vectors
    }
