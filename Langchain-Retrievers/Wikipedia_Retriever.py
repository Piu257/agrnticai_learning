from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
import os

load_dotenv()

wikipedia_retriever = WikipediaRetriever(top_k_results=2, lang='en')

docs= wikipedia_retriever.invoke(input='Rabindra Nath Tagore')

for i,doc in enumerate(docs):
    print(doc.page_content)

