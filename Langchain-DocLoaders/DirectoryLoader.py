from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = DirectoryLoader(path='books', glob='*.pdf', loader_cls=PyPDFLoader)
docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[326].page_content)
print(docs[326].metadata)

#for document in docs:
#    print(document.page_content)

docs1 = loader.lazy_load()
for document in docs1:
   print(document.page_content)

