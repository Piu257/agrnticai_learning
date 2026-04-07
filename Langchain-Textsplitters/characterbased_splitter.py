from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()
print(len(docs))

splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap = 0,
    separator=''
)

splits = splitter.split_documents(docs)
print(len(splits))
print(splits[0])