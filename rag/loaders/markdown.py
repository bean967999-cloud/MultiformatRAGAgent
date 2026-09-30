
from pathlib import Path
from langchain_core.documents import Document

from .base import BaseLoader

class MarkdownLoader(BaseLoader):
    def load(self,file_path:Path) -> list[Document]:
        text = file_path.read_text(encoding='utf-8')
        document = Document(
            page_content = text,
            metadata = {
                'source':str(file_path),
                'file_type':'markdown'
            }
        )

        return [document]

# V1 阶段暂时不需要解析 Markdown 的标题、表格、代码块。
