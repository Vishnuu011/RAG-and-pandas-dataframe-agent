import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 🔧 Import your custom classes
from server.app import (
    CreateMultiAgentSystem,
    CreateVectorEmbeddings,
    ApplyCharacterTextSplitter,
    LoadUnstructuredDocxFile,
    plotly_figures
)

# ✅ Initialize the multi-agent executor
executor = CreateMultiAgentSystem(
    dir_path=r"C:\Users\Vishnu\Desktop\RAG_Agent\Data-Analysis-Agent\data",
    document_loader=LoadUnstructuredDocxFile(),
    text_spliter=ApplyCharacterTextSplitter(),
    embeddings=CreateVectorEmbeddings()
).Setup_Agent_Executor_and_tools(
    df_path=r"C:\Users\Vishnu\Desktop\RAG_Agent\Data-Analysis-Agent\healthcare_stroke_dataset.csv"
)

# === Interactive loop ===
print("🤖 Multi-Agent Assistant Ready (type 'exit' to quit)\n")
while True:
    query = input("🧠 Ask: ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    try:
        plotly_figures.clear()  # clear previous plots

        response = executor.invoke({"input": query})
        print("\n🔍 Response:")
        print(response["output"])

        # ✅ Display plots if generated
        if plotly_figures:
            print("\n📊 Generated Plots:")
            for fig in plotly_figures:
                fig.show()

        # Optional: source docs
        if "source_documents" in response:
            print("\n📚 Source Docs:")
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"[Doc {i}] {doc.metadata.get('source', 'N/A')}")
                print(doc.page_content[:300] + "...\n")

    except Exception as e:
        print(f"❌ Error: {e}")
