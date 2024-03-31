from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader

class RAGVectorStore:
    '''Vector store Wrapper Class'''
    
    def __init__(self, data_dir: str, embeddings: object, store_type="FAISS"):
        self.data_dir = data_dir
        self.embeddings = embeddings
        self.store_type = store_type
        self.loader, self.docs, self.text_splitter, self.documents_split = None, None, None, None
    
    def create_loader(self, glob="**/*.txt", loader_cls=TextLoader):
        self.loader = DirectoryLoader(self.data_dir, glob=glob, loader_cls=loader_cls)
        
    def load_docs(self):
        self.docs = self.loader.load()
        
    def create_text_splitter(self, chunk_size=1000, chunk_overlap=0):
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
    def split_text(self, docs: list[object]) -> list[object]:
        self.documents_split = self.text_splitter.split_documents(docs)
    
    def get_db(self) -> object:
        if not self.loader or not self.docs or not self.text_splitter or not self.documents_split:
            raise Exception("""loader, text_splitter, docs, or documents_split is None.
                            Make sure to call methods for initializing these attributes before calling get_db.""")
        if self.store_type == "FAISS":
            return FAISS.from_documents(self.documents_split, self.embeddings)
        return None
