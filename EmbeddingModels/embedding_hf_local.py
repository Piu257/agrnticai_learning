import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))

documents = ["Delhi is the capital of India", "Mumbai is the financial capital of India"]
result11 = embedding.embed_documents(documents)

print(str(result11))
