from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]

with open('customer_agent_chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'query':'Where is my refund', 'chat_history':chat_history})

model = ChatOpenAI(model='gpt-4')

print(prompt)
print(model.invoke(prompt))
