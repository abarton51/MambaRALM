from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import DirectoryLoader
from src.config import device

class RAGVectorStore:
    '''Vector store Wrapper'''
    
    def __init__(self, data_dir: str, store_type="FAISS", chunk_size: int = 1000, chunk_overlap: int = 30):

        self.data_dir = data_dir
        self.store_type = store_type

        #document loader
        self.loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True, show_progress=True)
        
        #text splitter
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={
                'device': device
            }
        )
        
    def create_db(self, db_name : str, verbose : bool = False) -> object:

        docs = [x[0] for x in self.loader.load()]

        if verbose:
            
            print("Documents Loaded")

        documents_split = self.text_splitter.split_documents(docs)

        if verbose:
        
            print("Documents Split")

        print("Beginning Embeddings")

        if self.store_type == "FAISS":

            db = FAISS.from_documents(documents_split, self.embedding_model)

        db.save_local("vector_stores/{fn}".format(fn=db_name))
        
        return db

    def save_db(self, vector_store : VectorStore, db_name : str) -> None:

        vector_store.save_local("vector_stores/{fn}".format(fn=db_name))

    def load_db(self, db_name : str) -> VectorStore:

        if self.store_type == "FAISS":
        
            return FAISS.load_local("vector_stores/{fn}".format(fn=db_name), embeddings=self.embedding_model, allow_dangerous_deserialization=True)
