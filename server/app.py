import pandas as pd 
import os, sys

from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    Tool
)

from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_groq import ChatGroq
from langchain.prompts import (
    PromptTemplate, 
    MessagesPlaceholder
)
from typing import List, Optional
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import logging
from dotenv import load_dotenv

import warnings 
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")


"""
Initial creation of a RAG (Retrieval-Augmented Generation) pipeline for generating answers related to
Data Science, Machine Learning, Deep Learning, Computer Vision, and Data Analysis.
This pipeline uses RAG to enhance accuracy and context in responses.
"""

class LoadUnstructuredDocxFile:

    """
    A loader class to read unstructured DOCX files from a specified directory
    using LangChain's DirectoryLoader and UnstructuredFileLoader.
    """

    def __init__(self, dir_path: str):

        self.dir_path : str = dir_path
        self.glob : str = "**/*.docx"

    def Unstructured_Docx_FileLoader(self) -> List[Document] | None:

        """
        Loads DOCX (or other unstructured) files from a directory using the UnstructuredFileLoader.

        Returns:
            List[Document]: A list of documents loaded from the directory.
        """
        
        try:
            logger.info("Enter in Document Loader...")

            loader = DirectoryLoader(
                path=self.dir_path,
                glob=self.glob,
                loader_cls=UnstructuredFileLoader
            )
            documents = loader.load()

            logger.info("Exited from LoadUnstructuredDocxFile class....")
            return documents
        except Exception as e:
            logger.error(f"Error Occured in : {e}")



class ApplyCharacterTextSplitter:

    """
    Applies recursive character-based text splitting on a list of LangChain Document objects.
    Useful for breaking large documents into smaller chunks for better processing by language models.
    """

    def __init__(self):

        """
        Initializes the RecursiveCharacterTextSplitter with default chunk size and overlap.
        """

        try:
            self.text_spliter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap =200
            )
        except Exception as e:
            logger.error(f"Error Occured In : {e}")  

    def apply_text_spliter(self, documents: List[Document]) -> List[Document]:

        """
        Splits the input documents into smaller chunks using the initialized text splitter.

        Args:
            documents (List[Document]): A list of LangChain Document objects to split.

        Returns:
            List[Document]: A list of chunked Document objects.
        """

        try:
            logger.info("Entered In text spliter ....")

            texts = self.text_spliter.split_documents(documents)
            logger.info("Exited From apply text spliter ....")
            return texts
        except Exception as e:
            logger.error(f"Error Occured In : {e}") 
            return []              



class CreateVectorEmbeddings:

    """
    A utility class for creating text embeddings using HuggingFace's `all-MiniLM-L6-v2` model.
    Useful for converting text into vector representations for use in vector databases and retrieval.
    """

    def __init__(self):

        """
        Initializes the embedding class. Currently no arguments are required.
        """
        pass

    def Create_Embeddings(self) -> Optional[HuggingFaceEmbeddings]:

        """
        Creates and returns an instance of HuggingFaceEmbeddings using a pretrained model.

        Returns:
            Optional[Embeddings]: The embeddings model if successful, otherwise None.
        """

        try:
            logger.info("Entered Embedding Part ...")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Exited create Embedding...")
            return embeddings
        except Exception as e:
            logger.error(f"Error Occured In {e}")




class ChrmoVectorDataBase:

    """
    A wrapper class to manage a Chroma vector database for 
    storing and retrieving document embeddings.
    """

    def __init__(self):

        """
        Initializes the Chroma vector database interface.
        """

        self.Chromadb = Chroma

    def update_vector_store(
            self,
            text: List[Document],
            embeddings: Embeddings,
            persist_directory: str,
            collection_name: str
    ) -> Optional[Chroma]:
        
        """
        Creates or updates a Chroma vector store with the provided documents and embeddings.

        Args:
            text (List[Document]): A list of documents to embed and store.
            embeddings (Embeddings): The embedding model to use (e.g., HuggingFaceEmbeddings).
            persist_directory (str): Path to persist the vector store to disk.
            collection_name (str): Name of the vector store collection.

        Returns:
            Optional[Chroma]: The Chroma vector store object if successful, otherwise None.
        """

        try:
            logger.info("Entered In Vector db Creation...")
            vector_store = self.Chromadb.from_documents(
                documents=text,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )

            logger.info("Exited from update vector store ...")
            return vector_store
        except Exception as e:
            logger.error(f"Error Occured In {e}")    


class TemporaryPromptTemplate:

    """
    A class to define and return a custom prompt template for a retrieval-based assistant.
    """

    def __init__(self):

        """
        Initializes the prompt template with placeholders for context and question.
        """
        
        self.prompt_template = """
         You are a helpful assistant. Use the following context to answer the question.         

         If the answer is not directly found, try to infer it from the context.         

         If nothing is relevant, say "I couldn't find relevant information."         

         Context:
         {context}         

         Question: {question}
         Answer:"""

    def prompt_templates(self) -> Optional[PromptTemplate]:

        """
        Builds and returns a LangChain PromptTemplate object.

        Returns:
            Optional[PromptTemplate]: The compiled prompt template, or None if an error occurs.
        """

        try:
            PROMPT = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )
            return PROMPT
        except Exception as e:
            logger.error(f"Error Occured In {e}")  



class DocumentDocxRetrievalQAPipeline:

    """
    Builds a complete Retrieval-Augmented Generation (RAG) pipeline using:
    - ChromaDB vector store
    - HuggingFace embeddings
    - Groq language model
    - Custom prompt template
    """

    def __init__(
            self,
            text: List[Document],
            embeddings: Embeddings,
            persist_directory: str,
            collection_name: str,
            groq_model : str,
            search_type: str,
            chain_type: str,
            search_kwargs: dict,
            return_source_documents: bool,
            input_key: str,
    ):
        
        """
        Initializes the vector store, prompt template, and Groq model for RAG.
        """
        
        try:
            self.Groq_model = ChatGroq(
            model=groq_model,
            temperature=0.7,
            max_tokens=500
            )
        
            self.vectore_store = ChrmoVectorDataBase().update_vector_store(
                text=text,
                embeddings=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )

            if self.vectore_store is None:
                ValueError("Vectore Store Not Created...")


            self.PROMPT = TemporaryPromptTemplate().prompt_templates()

            if self.PROMPT is None:
                ValueError("Prompt Template Not Created")


            self.search_type = search_type
            self.chain_type = chain_type
            self.search_kwargs = search_kwargs
            self.return_source_documents = return_source_documents
            self.input_key = input_key
        except Exception as e:
            logger.error(f"Error Occured In : {e}")

    def get_retrieval_QA(self) -> Optional[RetrievalQA]:

        """
        Constructs and returns the RetrievalQA chain for answering user queries.

        Returns:
            Optional[RetrievalQA]: The complete RetrievalQA chain or None if an error occurs.
        """

        try:
            logger.info("Entered in retrieval qa ....")
            rag_chain = RetrievalQA.from_chain_type(
                llm = self.Groq_model,
                chain_type = self.chain_type,
                retriever = self.vectore_store.as_retriever(
                    search_type= self.search_type,
                    search_kwargs = self.search_kwargs
                ),
                chain_type_kwargs={"prompt": self.PROMPT},
                return_source_documents=self.return_source_documents,
                input_key=self.input_key          
            )

            logger.info("Exited from get retrieval qa ...")
            return rag_chain
        except Exception as e:
            logger.error(f"Error Occured In {e}")    


if __name__=='__main__':

    documents = LoadUnstructuredDocxFile(
        dir_path=r"C:\Users\Vishnu\Desktop\RAG_Agent\Data-Analysis-Agent\data"
    ).Unstructured_Docx_FileLoader()

    texts = ApplyCharacterTextSplitter().apply_text_spliter(
        documents=documents
    )
    
    embedding = CreateVectorEmbeddings().Create_Embeddings()

    qa_pipeline = DocumentDocxRetrievalQAPipeline(
    text=texts,
    embeddings=embedding,
    persist_directory="chroma-store",
    collection_name="ml_docs",
    groq_model="llama-3.3-70b-versatile",
    search_type="similarity",
    chain_type="stuff",
    search_kwargs={"k": 3},
    return_source_documents=True,
    input_key="query"
    )
    rag_chain = qa_pipeline.get_retrieval_QA()
    print("\nEnter queries about your PDFs (type 'exit' to quit)")
    while True:
        query = input("\nEnter Your Query : ")
        if query.lower() == "exit":
            break    

        result = rag_chain.invoke({"query": query})

        print(f"\nAnswer: {result['result']}")
