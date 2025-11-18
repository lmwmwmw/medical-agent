
from utils import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from glob import glob
import os
from utils import *


#定义分割器
def doc2vec():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap= 50,
    )
#读取分割文件
    dir_path = os.path.join(os.path.dirname(__file__), './data/input/')

    documents = []
    for file_path in glob(dir_path + '*.*'):
        loader = None
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path,encoding='utf-8')
        if file_path.endswith(file_path):
            loader = CSVLoader(file_path,encoding='utf-8-sig')
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        if loader:
            documents += loader.load_and_split(text_splitter)
    #将分割好的切片进行向量化存储
    if documents:
        vecdb = Chroma.from_documents(
            documents=documents,
            embedding=get_embeddings_model(),
            persist_directory=os.path.join(os.path.dirname(__file__), './data/db/'),
        )

if __name__ == '__main__':
    doc2vec()


