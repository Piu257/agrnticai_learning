from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
import os

load_dotenv()

# Sample documents
docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embeddings= OpenAIEmbeddings()

vectordb = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)


simple_retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={'k':5})
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(top_k_results=5, lambda_mult= 0.3),
    llm=ChatOpenAI()

)

query= 'How can I stay Healthy?'
#results= simple_retriever.invoke(query)
results = multiquery_retriever.invoke(query)

print(results)

