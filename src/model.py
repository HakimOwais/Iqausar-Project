from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.schemas import Response
from tools import retriever_tool
from utils import parse

from dotenv import load_dotenv
load_dotenv()

        
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that summarize the document around 1000 words."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(temperature=0)

llm_with_tools = llm.bind_functions([retriever_tool, Response])

agent = (
    {
        "input": lambda x: x["input"],
        # Format agent scratchpad from intermediate steps
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | parse
)


agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)

agent_executor.invoke(
    {"input": "Summarize the document please"},
    return_only_outputs=True,
)