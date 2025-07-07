from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from config import PERSIST_DIR

class RAGPipeline:
    def __init__(self, use_compression=False, chain_type="stuff"):
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.db = Chroma(embedding_function=self.embedding, persist_directory=PERSIST_DIR)
        self.base_retriever = self.db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.7})

        if use_compression:
            compressor = LLMChainExtractor.from_llm(self.llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
        else:
            self.retriever = self.base_retriever

        self.chat_history = []
        self.chain_type = chain_type
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type=chain_type
        )

    def add_pdf(self, pdf_path, orig_filename=None):
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        # Set metadata to original filename
        if orig_filename is not None:
            for d in docs:
                d.metadata["source"] = orig_filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        # Propagate metadata to chunks (if necessary)
        for c in chunks:
            c.metadata["source"] = orig_filename if orig_filename else pdf_path

        self.db.add_documents(chunks)
        return len(chunks)

    def ask(self, query):
        response = self.qa.invoke({
            "question": query,
            "chat_history": self.chat_history
        })
        self.chat_history.append((query, response["answer"]))
        return response