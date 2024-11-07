from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import dotenv
import chromadb

dotenv.load_dotenv()

paths = ['Food Security Nutrition 2023.pdf','Food Security Nutrition 2024.pdf']

class vectordb:
    def create_embeddings(self):
        self._load_pdf()
        self._store_into_chroma_db()

    def __init__(self, paths:list[str], embed_flag:bool = False):
        self._all_documents = []
        self.embedding=OpenAIEmbeddings(model='text-embedding-ada-002')
        self._paths = paths
        self.persistent_client = chromadb.PersistentClient()
        self.db = Chroma(
            collection_name="273_hackathon",
            embedding_function=self.embedding,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )

    def _load_pdf(self):
        if embed_flag:
            self.create_embeddings()

    def _load_pdf(self):
        # Load the PDF document
        for path in self._paths:
            pdf_loader = PyPDFLoader(path)
            raw_documents = pdf_loader.load()
            self._all_documents.extend(raw_documents)

    def _store_into_chroma_db(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(self._all_documents)
        self.vectorstore = Chroma.from_documents(collection_name='273_hackathon', documents=splits, embedding=self.embedding, persist_directory="./chroma_langchain_db")

obj = vectordb(paths)
vector_store = obj.db

#print(vector_store._collection.count())

#similarity = vector_store.similarity_search('food security for 2023')
#print(similarity)