# requests 用来向远程 Embedding API 发送 HTTP 请求。
import requests

# 导入我们之前定义的统一 Embedding 接口。
from rag.embeddings.base import BaseEmbedding



class APIEmbedding(BaseEmbedding):

    # 初始化API Embedding模型
    def __init__(

        self,
        # API 服务地址。
        # 后面真正使用时，从 .env 读取
        base_url:str,
        # API 密钥。
        api_key:str,
        # 模型名称。
        model:str,
    ):
        # 删除 base_url 最后多余的 "/"   比如：https://example.com/v1/
        # 会变成：https://example.com/v1
        # 这样后面拼接 /embeddings 时，
        # 就不会出现两个斜杠：
        # /v1//embeddings
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model


    #以下的embed_documents()和embed_query()方法  是比较基础的代码复用/抽取公共逻辑

    def embed_documents(
            self,
            texts:list[str]
    ) -> list[list[float]]:
        # 调用内部的 API 请求方法。
        # 这里把多个文本一次交给 API。

        data = self._request_embeddings(texts)

        # API 返回的数据通常是：
        # {
        #     "data": [
        #         {"embedding": [...]},
        #         {"embedding": [...]}
        #     ]
        # }
        # 把每一条 embedding 取出来。
        vectors = [
            item["embedding"]
            for item in sorted(data["data"], key=lambda item: item["index"])
        ]

        if len(vectors) != len(texts):
            raise ValueError("Embedding API 返回的向量数量与输入文本数量不一致")

        return vectors

    def embed_query(
            self,
            text: str
    ) -> list[float]:

        data = self._request_embeddings([text])

        return data["data"][0]["embedding"]

    #真正发送HTTP请求
    # embed_documents()
    # embed_query()
    # 这样可以避免把相同的 HTTP 请求代码复制两遍
    # 至于为什么前面有_这是一种Python中非常常见的命名习惯，这个方法主要供类内部使用
    def _request_embeddings(
            self,
            texts: list[str]
    ) -> dict:

        # 构造 API 请求地址
        # https://example.com/v1/embeddings
        url = f"{self.base_url}/embeddings"

        # 设置 HTTP 请求头
        headers = {
            # Bearer Token 是很多 API 的常见认证方式
            "Authorization": f"Bearer {self.api_key}",

            # 我发送的是 JSON 数据
            "Content-Type": "application/json"
        }

        # 构造请求体
        # 这是很多 OpenAI-compatible Embedding API
        # 常见的数据结构
        payload = {
            #使用哪个Embedding模型
            "model": self.model,

            # 要进行 Embedding 的文本
            "input": texts
        }

        # 发送 POST 请求
        response = requests.post(
            url,

            #请求头
            headers=headers,

            #requests会自动把Python字典转换成json
            json=payload,

            # 最多等待 60 秒。
            # 防止 API 一直没有响应，
            # 程序永远卡住。
            # 这属于非常基础的工程化处理
            timeout=60
        )

        # 401 → API Key 不正确
        # 404 → 地址不正确
        # 500 → 服务端错误
        response.raise_for_status()

        # 将 API 返回的 JSON 字符串
        # 转换成 Python 字典。
        return response.json()
