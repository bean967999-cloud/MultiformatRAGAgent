#负责加载本地 BGE-M3，并提供“批量文本向量化”和“单个查询向量化”两个能力

# SentenceTransformer这个类负责加载和运行类似 BGE-M3 这样的 Embedding 模型。
from sentence_transformers import SentenceTransformer

# 导入我们自己定义的 BaseEmbedding。
# LocalBGEEmbedding 后面要继承它，
# 这样本地模型和以后 API 模型就可以使用统一的接口。
from rag.embeddings.base import BaseEmbedding

class LocalBGEEmbedding(BaseEmbedding):

    '''
    device:指定要加载的Embedding模型
    默认使用BGE-M3
    cuda:指定要使用GPU
    '''

    # __init__ 是创建对象时自动执行的方法。
    # 这里主要负责：加载 BGE-M3 模型。

    def __init__(
            self,
            model_name_or_path: str = "BAAI/bge-m3",
            device: str = "cuda"
    ):
        self.model = SentenceTransformer(
            model_name_or_path,
            device = device
        )

    # RAG 的“入库阶段”
    def embed_documents(
            self,
            texts: list[str]    #一次处理多个文本,将多个chunk一次批量送进去，必须是list[str]
    ) -> list[list[float]]:     #向量的包装


        # normalize_embeddings=True：
        #   对生成的向量进行归一化，
        #   方便后面进行向量相似度计算。
        # 把texts里面的所有文本交给BGE - M3，让模型把它们分别转换成向量，并且把这些向量进行归一化
        #   数学上叫单位向量归一化，向量的模为1
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        # encode()  返回的是 NumPy 数组。
        # 后面的程序、数据库或者接口处理时，
        # Python 原生 list 通常更加方便
        # 所以使用 tolist() 把 NumPy 数组转换成 Python list。
        return vectors.tolist()

    def embed_query(
            self,
            text:str
    ) -> list[float]:     #查询向量的包装，查询通常是一句话，所以返回的是一个list[float]

        '''
        把text里面的所有文本交给BGE - M3，让模型把它们分别转换成向量，并且把这些向量进行归一化
        两个已经归一化的向量，它们的点积，就等价于余弦相似度,因为归一化后，向量的模为1
        余弦相似度就是向量之间的角度角，范围在[-1,1]之间，1表示完全相似，-1表示完全相反，0表示不相关
        余弦相似度本来是：
                        A·B / ∣A∣∣B∣
        '''



        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        # 将 NumPy ndarray转换成 Python list
        #转化后方便后续的JSON、数据库、API、序列化
        return vector.tolist()
