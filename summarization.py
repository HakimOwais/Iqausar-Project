from langchain.chains import AnalyzeDocumentChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


from dotenv import load_dotenv

load_dotenv()

# Load in document to retrieve over
# loader = UnstructuredPDFLoader("RFP_MAPP10022014.pdf")
# documents = loader.load()

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from IPython.display import display, Markdown


pdfreader = PdfReader('RFP_MAPP10022014.pdf')

from typing_extensions import Concatenate
# reading the text from pdf
text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        text += content

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

# pdfreader.topics

## Splittting the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
chunks = text_splitter.create_documents([text])

# prompt="""
# Please summarize the below speech:
# Speech:`{text}'
# Summary:
# """

# Define prompt for summarization of each chunk
prompt = """<bos><start_of_turn>user
Summarize the following text in a technical way. Focus on facts, numbers and strategies used. 
Divide the summary in chapters, be impersonal and use bullet points:

{text}<end_of_turn>
<start_of_turn>model"""

map_prompt_template=PromptTemplate(input_variables=['text'],
                                    template=prompt)


# final_combine_prompt='''
# Provide a final summary of the entire speech with these important points.
# Add a Generic Motivational Title,
# Start the precise summary with an introduction and provide the
# summary in bullet points for the speech in about 2000 words only.
# Speech: `{text}`
# '''

final_combine_prompt = """<bos><start_of_turn>user
You are given a text containing summaries of different part of a document.
Create one single summary combining all the information of the chapters. Divide the summary in chapters, be impersonal and use bullet points:

{text}<end_of_turn>
<start_of_turn>model"""

final_combine_prompt_template=PromptTemplate(input_variables=['text'],
                                             template=final_combine_prompt)

summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt_template,
    combine_prompt=final_combine_prompt_template,
    verbose=False
)
out_summary = summary_chain.invoke(chunks)


# display(Markdown(out_summary['output_text'].replace('#', '')))
print(out_summary)