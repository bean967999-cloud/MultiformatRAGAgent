
from pathlib import Path

from langchain_core.documents import Document


class BaseLoader:

    def load(self,file_path:Path) ->list[Document]:
        raise NotImplementedError
