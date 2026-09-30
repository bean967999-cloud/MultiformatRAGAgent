

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    '''
    要做本地BGE-M3和API Embedding两种方式，
    所以要定义统一的接口
    embed_documents()
    embed_query()

    后面的 embedding_node 只需要调用这个接口，
    不需要关心底层到底使用本地模型还是 API。
    '''

    @abstractmethod
    def embed_documents(
            self,
            documents: list[str]
    ) -> list[list[float]]:
        '''
        参数：
            documents:
                多个文本组成的列表。
        返回：
            每个文本对应一个向量。
        这个方法主要用于：
            RAG 入库阶段
        因为我们有很多 Chunk，
        所以需要批量把 Chunk 转换成向量。
        '''

        pass


    #将问题也向量化处理,用于之后的向量检索
    @abstractmethod
    def embed_query(
            self,
            query: str
    ) -> list[float]:
        pass
