from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = CSVLoader(file_path='Social_Network_Ads.csv',encoding='utf-8')
docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)
