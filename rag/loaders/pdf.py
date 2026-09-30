from pathlib import Path

from pymupdf import open  as open_pdf
from langchain_core.documents import Document
#为什么我们一定要有 Document？
#这就是我们现在这个多格式 RAG 项目最核心的设计思想
from .base import BaseLoader

class PDFLoader(BaseLoader):
    def load(self,file_path:Path) ->list[Document]:
        pdf = open_pdf(file_path)       #打开 PDF 文件
        documents = []          #用来存储所有的文档，先建立好这个列表，后面会往里面添加文档对象
        for page_number,page in enumerate(pdf):     #遍历的时候，同时获取编号,遍历所有的页面，同时给每个元素一个编号
            text = page.get_text().strip()      #把当前 PDF 页面的文字提取出来，去掉前后的空白字符

            if text:
                #调用LangChain的Document类，创建一个对象
                #创建一个 Document 对象，并把当前 PDF 页面的内容和相关信息放进去
                document = Document(
                    page_content=text,                  #page_content是真正的文本内容
                    metadata={                          #metadata是元数据，记录当前文档的相关信息
                        "source": str(file_path),       #记录当前文档的来源文件路径
                        "file_type": "pdf",
                        "page": page_number + 1,        #py里面从0开始，所以要+1，才变成第一页的页码是1
                    }
                )
                documents.append(document)              #把当前文档对象添加到 documents 列表中


        # # for 循环结束以后，再关闭 PDF
        pdf.close()                                 #关闭 PDF 文件

        # for 循环结束以后，再返回所有 Document
        return documents                            #返回所有文档对象的列表
