
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load in document to retrieve over
loader = UnstructuredPDFLoader("storytelling-with-data-cole-nussbaumer-knaflic.pdf")
documents = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Here is where we add in the fake source information
for i, doc in enumerate(texts):
    doc.metadata["page_chunk"] = i

# Create our retriever
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
retriever = vectorstore.as_retriever()


retriever_tool = create_retriever_tool(
    retriever,
    "summary-retriever-tool",
    "Retrieve the summary of the document",
)
