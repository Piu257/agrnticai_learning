from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

chat_prompt_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} assistant'),
    ('human','tell me about {topic}')
])

domain='Health'
topic='Obesity'
final_prompt = chat_prompt_template.invoke({'domain': domain, 'topic':topic})

print(model.invoke(final_prompt).content)

