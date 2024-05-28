# from langchain_community.document_loaders import UnstructuredPDFLoader


# loader = UnstructuredPDFLoader("storytelling-with-data-cole-nussbaumer-knaflic.pdf")

# data = loader.load()

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field

import json

from langchain_core.agents import AgentActionMessageLog, AgentFinish

from dotenv import load_dotenv

load_dotenv()

# Load in document to retrieve over
loader = UnstructuredPDFLoader("RFP_MAPP10022014.pdf")
documents = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Here is where we add in the fake source information
for i, doc in enumerate(texts):
    doc.metadata["page_chunk"] = i

# Create our retriever
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
retriever = vectorstore.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "summary-retriever-tool",
    "Retrieve the summary of the document",
)




class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )
    
def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )
        
        
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that summarize the document in 1000 words."),
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
    {"input": "What is the document about and what are its contents and summarize each content"},
    return_only_outputs=True,
)