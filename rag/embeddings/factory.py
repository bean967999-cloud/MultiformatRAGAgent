
import os
from functools import lru_cache
from dotenv import load_dotenv
#两种Embedding模型
from rag.embeddings.local import LocalBGEEmbedding
from rag.embeddings.api import APIEmbedding

# 加载 .env 文件中的环境变量
load_dotenv()

class EmbeddingFactory:

    @staticmethod
    @lru_cache(maxsize=2)
    def _local(model_path: str, device: str):
        return LocalBGEEmbedding(model_name_or_path=model_path, device=device)

    @staticmethod
    def create():

        #读取当前使用哪一种加载 Embedding Provider
        #默认使用local
        provider = os.getenv(
            "EMBEDDING_PROVIDER",
            "local"
        ).lower()


        if provider == "local":

            model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH")

            if not model_path:
                raise ValueError(
                    "缺少 LOCAL_EMBEDDING_MODEL_PATH"
                )

            return EmbeddingFactory._local(model_path, os.getenv("LOCAL_EMBEDDING_DEVICE", "cuda"))


        elif provider == "api":

            base_url = os.getenv("EMBEDDING_BASE_URL")

            api_key = os.getenv("EMBEDDING_API_KEY")

            model = os.getenv("EMBEDDING_MODEL")

            if not base_url:
                raise ValueError(
                    "缺少 EMBEDDING_BASE_URL"
                )
            if not api_key:
                raise ValueError(
                    "缺少 EMBEDDING_API_KEY"
                )
            if not model:
                raise ValueError(
                    "缺少 EMBEDDING_MODEL"
                )

            return APIEmbedding(
                base_url = base_url,
                api_key = api_key,
                model = model
            )

        else:
            raise ValueError(
                f"不支持的 EMBEDDING_PROVIDER: {provider}"
            )
