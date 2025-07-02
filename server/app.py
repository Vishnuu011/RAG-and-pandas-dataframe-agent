import pandas as pd 
import os, sys
import uuid

import io
import contextlib

from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
    Tool
)

from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_groq import ChatGroq
from langchain.prompts import (
    ChatPromptTemplate, 
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
import seaborn as sns 
import matplotlib.pyplot as plt

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

    def __init__(self):
        self.glob : str = "**/*.docx"

        self.show_progress = True

    def Unstructured_Docx_FileLoader(self, dir_path: str) -> List[Document] | None:

        """
        Loads DOCX (or other unstructured) files from a directory using the UnstructuredFileLoader.

        Returns:
            List[Document]: A list of documents loaded from the directory.
        """
        
        try:
            logger.info("Enter in Document Loader...")

            loader = DirectoryLoader(
                path=dir_path,
                glob=self.glob,
                loader_cls=UnstructuredFileLoader,
                show_progress=self.show_progress
            )
            documents = loader.load()

            logger.info("Exited from LoadUnstructuredDocxFile class....")
            return documents
        except Exception as e:
            logger.error(
                "Failed to initialize Unstructured_Docx_FileLoader",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            ) 
            raise



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
            logger.error(
                "Failed to initialize RecursiveCharacterTextSplitter",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            ) 
            raise

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
            logger.error(
                "Failed to initialize apply_text_splitter",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            ) 
            #raise
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
            logger.error(
                "Failed to initialize Create_Embeddings",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            ) 
            raise




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
            logger.error(
                "Failed to initialize update_vector_store",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            ) 
            raise



class RAGPromptTemplate:

    """
    A class to define and return a RAG-style ChatPromptTemplate
    for use in retrieval-augmented generation pipelines.
    """


    def __init__(self):

        """
        Initializes the RAG-style chat prompt with system, human,
        chat history, and agent scratchpad placeholders.
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a RAG-powered data science and AI assistant. Use tools when needed to gather or compute information. "
             "If the answer is not directly found, try to infer it. If nothing is relevant, say 'I couldn't find relevant information.'"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),  
        ])

    def get_prompt_template(self) -> Optional[ChatPromptTemplate]:

        """
        Returns the constructed ChatPromptTemplate for RAG-style interactions.

        Returns:
            Optional[ChatPromptTemplate]: The compiled prompt template, or None if an error occurs.
        """

        try:
            return self.prompt_template
        except Exception as e:
            print(f"Error in RAGPromptTemplate: {e}")
            return None



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


            self.PROMPT = RAGPromptTemplate().get_prompt_template()

            if self.PROMPT is None:
                ValueError("Prompt Template Not Created")


            self.search_type = search_type
            self.chain_type = chain_type
            self.search_kwargs = search_kwargs
            self.return_source_documents = return_source_documents
            self.input_key = input_key

        except Exception as e:
            logger.error(
                "Failed to initialize DocumentDocxRetrievalQAPipeline",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            ) 
            raise


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
            logger.error(
                "Failed to initialize get_retrieval_QA",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            ) 
            raise



class CreateSharedMemoryBuffer:

    """
    A class to create and manage a shared ConversationBufferMemory instance.

    Attributes:
        memory_key (str): The key used to store conversation memory.
        return_messages (bool): Whether to return messages instead of string.
        output_key (str): The key that corresponds to the output in memory.
    """

    def __init__(
            self,
            memory_key: str,
            return_messages: bool,
            output_key: str
    ) -> None:
        
        """
        Initializes the shared memory buffer configuration.

        Args:
            memory_key (str): Key for memory storage.
            return_messages (bool): Flag to return messages or not.
            output_key (str): Key for the memory output.
        """

        self.memory_key = memory_key
        self.return_messages = return_messages
        self.output_key = output_key

    def Conversation_Buffer_Memory(self) -> ConversationBufferMemory:

        """
        Creates and returns a ConversationBufferMemory instance based on configuration.

        Returns:
            ConversationBufferMemory: The memory instance for storing conversation history.

        Raises:
            Logs the error if creation fails.
        """

        try:
            memory = ConversationBufferMemory(
                memory_key=self.memory_key,
                return_messages=self.return_messages,
                output_key=self.output_key
            )

            return memory
        except Exception as e:
            logger.error(
                "Failed to initialize ConversationBufferMemory",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            )
            raise

class MultiAgentPromptTemplate:

    """
    Class to generate a reusable multi-agent system prompt template.
    This template includes memory context, human input, and scratchpad for tool-based reasoning.
    """

    def __init__(self):

        """
        Initializes the multi-agent chat prompt with system instructions and message placeholders.
        """

        try:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a multi-agent assistant. Use RAGTool for document-based questions. \n"
                "Use DataAnalysisTool for access to a pandas DataFrame, data analysis, data engineering and preprocessing and machine learning."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        except Exception as e:
            logger.error(
                "Failed to initialize MultiAgentPromptTemplate",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            )
            raise
               

    def get_multiagent_chat_template(self) -> Optional[ChatPromptTemplate]:

        """
        Returns:
            Optional[ChatPromptTemplate]: The compiled chat prompt template for multi-agent interaction.
        """

        try:
            prompt = self.prompt
            return prompt
        except Exception as e:
            logger.error(
                "Failed to initialize MultiAgentPromptTemplate",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            )
            raise
   

class CreateMultiAgentSystem:

    """
    Class to build and return a fully functional multi-agent system using
    document RAG + CSV analysis with shared memory.
    """
    
    def __init__(
            self,
            dir_path: str,
            document_loader: LoadUnstructuredDocxFile,
            text_spliter: ApplyCharacterTextSplitter,
            embeddings: CreateVectorEmbeddings

    ):
        
        """
        Initializes the system with document and embedding configuration.

        Args:
            dir_path (str): Path to the document folder.
            document_loader (LoadUnstructuredDocxFile): Loader class to extract documents.
            text_spliter (ApplyCharacterTextSplitter): Splits documents into chunks.
            embeddings (CreateVectorEmbeddings): Embedding model interface.
        """
        
        try:
            self.document_loader = document_loader
            self.dir_path = dir_path
            self.texts = text_spliter
            self.embedding = embeddings
            self.shared_memory = CreateSharedMemoryBuffer(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize CreateMultiAgentSystem",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            )
            raise 


    def Setup_Agent_Executor_and_tools(self, df_path: str) -> AgentExecutor:

        """
        Constructs the multi-agent system and returns an AgentExecutor.

        Args:
            csv_dataframes (Dict[str, pd.DataFrame]): Dictionary of loaded CSVs.

        Returns:
            AgentExecutor: Fully wired LangChain multi-agent executor with memory.
        """

        try:
            document = self.document_loader.Unstructured_Docx_FileLoader(
                dir_path=self.dir_path
            )

            text = self.texts.apply_text_spliter(
                documents=document
            )

            embeddings = self.embedding.Create_Embeddings()

            rag_pipeline = DocumentDocxRetrievalQAPipeline(
                text=text,
                embeddings=embeddings,
                persist_directory="chroma-store",
                collection_name="ml_docs",
                groq_model="llama-3.3-70b-versatile",
                search_type="similarity",
                chain_type="stuff",
                search_kwargs={"k": 3},
                return_source_documents=True,
                input_key="question"
            )

            rag_chain = rag_pipeline.get_retrieval_QA()
            

            rag_tool = Tool(
                name="RAGTool",
                func=lambda x: rag_chain({"question": x})["result"],
                description="Answer questions using document knowledge base."
            )
            df = pd.read_csv(df_path)

            # Fixed: Create REPL tool with proper locals
            python_tool = PythonAstREPLTool(locals={'df': df})

            def run_and_return_output_or_plot(code: str):
                try:
                    local_vars = {'df': df, 'plt': plt}
                    result = None

                    # --- Capture print output ---
                    output_buffer = io.StringIO()
                    with contextlib.redirect_stdout(output_buffer):
                        # Try to eval first (for expressions like df.describe())
                        if "\n" not in code and not code.strip().endswith(";") and not code.strip().startswith("print"):
                            result = eval(code, {}, local_vars)
                            if result is not None:
                                print(result)
                        else:
                            exec(code, {}, local_vars)

                    stdout_output = output_buffer.getvalue()

                    # --- Check for plot ---
                    fig = plt.gcf()
                    if fig.get_axes():
                        filename = f"plot_{uuid.uuid4().hex[:8]}.png"
                        plot_dir = os.path.join("backend", "plots")
                        os.makedirs(plot_dir, exist_ok=True)
                        filepath = os.path.join(plot_dir, filename)
                        plt.savefig(filepath)
                        plt.close(fig)
                        return f"📊 Plot saved at: {filepath}\n\n📝 Insights:\n{stdout_output.strip()}"

                    # --- No plot, only insights ---
                    if stdout_output.strip():
                        return f"📝 Insights:\n{stdout_output.strip()}"

                    return "✅ Code ran successfully, but no output or plot returned."

                except Exception as e:
                    return f"❌ Error while running code: {e}"   

            
            analysis_tool = Tool.from_function(
                name="DataAnalysisTool",
                func=run_and_return_output_or_plot,
                description=(
                    "Use this tool to perform data analysis and EDA on the DataFrame `df` using pandas, matplotlib, and seaborn.\n"
                    "Supports:\n"
                    "- Plotting: histogram, boxplot, barplot, pie chart, countplot (e.g., df.hist(), df['col'].plot(kind='box'), df.hist(), sns.boxplot(...))\n"
                    "- Descriptive stats: df.describe(), df.info(), df.skew(), df.isnull().sum(), df.value_counts()\n"
                    "- Grouping, filtering, sorting, merging, renaming, and missing value handling\n"
                    "- Correlation analysis: df.corr(), heatmaps\n"
                    "- Outlier detection using IQR method:\n"
                    "    Q1 = df['col'].quantile(0.25)\n"
                    "    Q3 = df['col'].quantile(0.75)\n"
                    "    IQR = Q3 - Q1\n"
                    "    outliers = df[(df['col'] < Q1 - 1.5 * IQR) | (df['col'] > Q3 + 1.5 * IQR)]\n"
                    "- Insight generation: print summaries, trends, patterns in the data\n"
                    "Input must be valid pandas/matplotlib code. Returns textual insights or saved plot paths."
                )
            )

            shared_memory = self.shared_memory.Conversation_Buffer_Memory()

            tools = [rag_tool, analysis_tool]

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=500
            )

            prompts = MultiAgentPromptTemplate().get_multiagent_chat_template()

            create_agent = create_tool_calling_agent(
                llm=llm,
                tools=tools,
                prompt=prompts
            )

            return AgentExecutor(
                agent=create_agent,
                tools=tools,
                memory=shared_memory,
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )



        except Exception as e:
            logger.error(
                "Failed to initialize Setup_Agent_Executor_and_tools",
                exc_info=True,
                extra={
                    "error_type" : type(e).__name__,
                    "error_message" : str(e)
                }
            )
            raise


        

       