
from pathlib import Path
from rag.loaders.factory import LoaderFactory

class DocumentLoaderService:

    @staticmethod
    def load(file_path: Path):      #DocumentLoaderService 对外提供一个“加载文档”的入口
        loader = LoaderFactory.get_loader(file_path)    #让 Factory 找到的那个具体 Loader，真正执行文件读取

        documents = loader.load(file_path)

        return documents
