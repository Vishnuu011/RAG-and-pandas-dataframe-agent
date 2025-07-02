from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_groq import ChatGroq
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sklearn

# 🔁 plot storage (for Streamlit etc.)
plotly_figures = []
def display(fig):
    plotly_figures.append(fig)
    return fig

# ✅ Load CSV
df = pd.read_csv("healthcare_stroke_dataset.csv")

# ✅ Groq LLM
llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)


python_repl = PythonAstREPLTool()
python_repl.name = "python_repl_ast"
python_repl.description = (
    "Executes Python code for data analysis with these constraints:\n"
    "1. ONLY use pandas, sklearn, and plotly\n"
    "2. Data is pre-loaded as `df`\n"
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


# ✅ Create agent (do NOT pass extra_tools, do enable dangerous code)
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    handle_parsing_errors=True,
    agent_type="tool-calling",
    extra_tools=[python_repl],
    allow_dangerous_code=True
)


# ✅ Test prompt
response = agent.invoke({"input": "Group and compare the age and work_type, smoking_status write a brief explanation around 100 words."})
print("📊 Output:\n", response["output"])