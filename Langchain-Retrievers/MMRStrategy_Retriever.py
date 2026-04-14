from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embeddings= OpenAIEmbeddings()

vectordb = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)


simple_retriever = vectordb.as_retriever(search_type='similarity', top_k_results=3)
mmr_retriever = vectordb.as_retriever(search_type='mmr', top_k_results=3, lambda_mult= 0.3)

query= 'What is LangChain?'
#results = simple_retriever.invoke(query)

#print(results)

#for i,doc in enumerate(results) : 
#    print(doc.page_content)

mmr_results = mmr_retriever.invoke(query)
print(mmr_results)
#for i,doc in enumerate(mmr_results) : 
#    print(doc.page_content)
