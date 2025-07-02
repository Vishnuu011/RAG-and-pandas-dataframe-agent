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
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
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
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import sklearn
from IPython.display import display
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
            temperature=0.3,
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
                (
                    "system",
                    """You are a Coordinator Agent managing two tools:
1. RAGTool - Answers questions using documents.
2. DataAnalysisTool - Executes Python code to analyze a pandas DataFrame preloaded as `df`.

🧠 Guidelines:
- You MUST use the DataAnalysisTool to compute all outputs from the DataFrame.
- NEVER guess or hallucinate any values — only analyze `df` using real code (e.g., df.shape, df.describe(), groupby, etc).
- Use `print(...)` to output analysis results.
- For visualizations, use `display(fig)` with seaborn or plotly.
- Python variables persist between calls.

📝 Special Instructions for summaries:
If asked to "analyze and explain" or "summarize in 100 words":
→ FIRST call DataAnalysisTool to analyze `df` (e.g., print(df.describe()), print(df.info()), or groupby stats).
→ THEN summarize the findings in ~100 words using real observed values.

📌 Example:
User: Analyze the DataFrame and write a brief 100-word report.

Assistant:
<tool_call>DataAnalysisTool{{"input_str": "print(df.describe()); print(df.info())"}}</tool_call>
[Then write the summary based on what you saw.]

Start your reasoning using tools. Do not skip the Python step.
"""
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        except Exception as e:
            logger.error(
                "Failed to initialize MultiAgentPromptTemplate",
                exc_info=True,
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise

    def get_multiagent_chat_template(self) -> Optional[ChatPromptTemplate]:
        """
        Returns:
            Optional[ChatPromptTemplate]: The compiled chat prompt template for multi-agent interaction.
        """
        try:
            return self.prompt
        except Exception as e:
            logger.error(
                "Failed to return prompt from MultiAgentPromptTemplate",
                exc_info=True,
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise


plotly_figures = []

def display(fig):
    plotly_figures.append(fig)
    return fig

import builtins
builtins.display = display   

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
                groq_model="llama3-70b-8192",
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

       
            python_repl = PythonAstREPLTool()
            python_repl.name = "python_repl_ast"
            python_repl.description = (
                "Executes Python code for data analysis with these constraints:\n"
                "1. ONLY use pandas, sklearn, and plotly\n"
                "2. Data is pre-loaded as df\n"
                "3. Use display(fig) to store or view plots\n"
                "4. Use print() to show outputs\n"
                "5. Variables persist between executions"
            )
            python_repl.locals.update({
                "df": df,
                "pd": pd,
                "sklearn": sklearn,
                "px": px,
                "go": go,
                "display": display
            })

         
            df_llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)

            # ✅ Create Pandas Agent
            df_agent = create_pandas_dataframe_agent(
                llm=df_llm,
                df=df,
                verbose=True,
                handle_parsing_errors=True,
                extra_tools=[python_repl],
                agent_type="tool-calling",
                allow_dangerous_code=True
            )


            @tool
            def DataAnalysisTool(input_str: str) -> str:
                """Use for data analysis, visualization, and modeling on DataFrame df."""
                return df_agent.invoke({"input": input_str})["output"]

 
            tools = [rag_tool, DataAnalysisTool]

    
            shared_memory = self.shared_memory.Conversation_Buffer_Memory()


            llm = ChatGroq(model="llama3-70b-8192", temperature=0.3, max_tokens=500)

        
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


        

       