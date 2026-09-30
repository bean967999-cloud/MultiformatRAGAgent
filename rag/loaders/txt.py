from pathlib import Path

from langchain_core.documents import Document
from .base import BaseLoader

class TXTLoader(BaseLoader):
    def load(self,file_path:Path) ->list[Document]:
        text = file_path.read_text(encoding='utf-8')

        document = Document(
            page_content = text,
            metadata = {
                'source':str(file_path),
                'file_type':'txt'
            }
        )
        return [document]
