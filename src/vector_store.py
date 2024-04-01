from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader

class RAGVectorStore:
    '''Vector store Wrapper Class'''
    
    def __init__(self, data_dir: str, embeddings: object, store_type="FAISS", chunk_size: int = 1000, chunk_overlap: int = 30):

        self.data_dir = data_dir
        self.embeddings = embeddings
        self.store_type = store_type
        self.loader, self.docs, self.text_splitter, self.documents_split = None, None, None, None

        #document loader
        self.loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True, show_progress=True)
        
        #text splitter
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
    def get_db(self) -> object:

        docs = self.loader.load()
        documents_split = self.text_splitter.split_documents(docs)

        if self.store_type == "FAISS":
            return FAISS.from_documents(documents_split, self.embeddings)
        
        return None
