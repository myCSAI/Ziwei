import sys
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from langchain.document_loaders import PyPDFLoader

sys.path.append('../..')
openai.api_key = "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"
#需要导入的书籍的pdf版本
path = "/home/admin1/桌面/data_pdf"
embedding_function = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")


def build_chromadb():
    if path:
        loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pages = loader.load()
        # print(len(pages))
        # print(page.page_content[0:500])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=10
        )
        splits = text_splitter.split_documents(pages)
        chroma = Chroma.from_documents(splits, embedding_function, persist_directory="/home/admin1/桌面/chromaDb")
        chroma.persist()


if __name__ == '__main__':
    build_chromadb()
    db = Chroma(persist_directory="/home/admin1/桌面/chromaDb", embedding_function=embedding_function)
    docs = db.similarity_search("火命", k=1)
    print(docs)
