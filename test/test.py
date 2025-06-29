from server.app import *

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
    input_key="question"
    )
    rag_chain = qa_pipeline.get_retrieval_QA()
    print("\nEnter queries about your PDFs (type 'exit' to quit)")
    while True:
        query = input("\nEnter Your Query : ")
        if query.lower() == "exit":
            break    

        result = rag_chain.invoke({"question": query})

        print(f"\nAnswer: {result['result']}")
