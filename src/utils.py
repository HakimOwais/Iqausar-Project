import json

from langchain_core.agents import AgentActionMessageLog, AgentFinish

from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def load_documents(file_path: str):
    """
    Load a document from the specified file path using UnstructuredPDFLoader.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        documents: The loaded documents.
    """
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()

def split_document_into_chunks(documents, chunk_size=1000, chunk_overlap=0):
    """
    Split the documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: The documents to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        texts: The split chunks of the documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def add_metadata_to_chunks(chunks):
    """
    Add metadata to each chunk indicating its page chunk index.

    Args:
        chunks: The document chunks.

    Returns:
        chunks: The document chunks with added metadata.
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata["page_chunk"] = i
    return chunks

def create_retriever(chunks, collection_name="state-of-union"):
    """
    Create a retriever from the document chunks.

    Args:
        chunks: The document chunks.
        collection_name (str): The name of the collection to store in Chroma.

    Returns:
        retriever: The created retriever.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name=collection_name)
    response = vectorstore.as_retriever()
    return response

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