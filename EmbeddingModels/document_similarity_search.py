from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents=['Poushali is good girl', 'Ram is God', 'Mother is kind']
doc_embeddings = embeddings.embed_documents(documents)
query= 'Tell me about Ram'
query_embedding=embeddings.embed_query(query)

##cosine similarity search works on 2D array 
scores = cosine_similarity([query_embedding],doc_embeddings)[0]

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(documents[index])

