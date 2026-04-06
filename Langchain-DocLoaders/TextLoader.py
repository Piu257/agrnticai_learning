from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = TextLoader('cricket.txt',encoding='utf-8')
docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0])
print(docs[0].page_content)
print(docs[0].metadata)


model = ChatOpenAI()

prompt = PromptTemplate(
    template='Write summary of the poem {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'poem':docs[0].page_content})
print(result)


