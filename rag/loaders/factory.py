'''
工厂模式：不要让调用者知道对象怎么创建。
Factory:
loader = LoaderFactory.get_loader("test.pdf")
调用者只关心：
给我一个能读取这个文件的 Loader。
'''

from pathlib import Path

from .txt import TXTLoader
from .docx import DOCXLoader
from .markdown import MarkdownLoader
from .pdf import PDFLoader
from .ppt import PPTLoader
from .excel import ExcelLoader


class LoaderFactory:
    '''
    普通类型：需要实例化对象
    factory = LoaderFactory()
    factory.get_loader()

    工厂类型：不需要实例化对象，直接调用类方法即可
    一般情况下，工厂模式的类，都是静态方法
    这个方法属于类 但是：不需要创建对象，静态方法因为： LoaderFactory本身没有状态
    '''

    @staticmethod
    def get_loader(file_path:Path):

        #file_path.suffix 获取后缀    .lower()  test.PDF和test.pdf的后缀都是.pdf 统一为小写
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            return TXTLoader()

        elif suffix == ".docx":
            return DOCXLoader()

        elif suffix in [".md", ".markdown"]:
            return MarkdownLoader()

        elif suffix == ".pdf":
            return PDFLoader()

        elif suffix == ".pptx":
            return PPTLoader()

        elif suffix == ".xlsx":
            return ExcelLoader()

        else:
            raise ValueError(f"不支持的文件类型：{suffix}")
