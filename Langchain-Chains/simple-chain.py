import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm_endpoint = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-Coder-Next',
    task='text-generation',
    huggingfacehub_api_token=api_key
)

model = ChatHuggingFace(llm= llm_endpoint)

prompt = PromptTemplate(
    template="""
Give 5 interesting facts about {topic}.
""",
input_variables=['topic']
)

parser = StrOutputParser()

chain  = prompt | model | parser

result = chain.invoke({'topic':'After Death'})

print(result)

chain.get_graph().print_ascii()

