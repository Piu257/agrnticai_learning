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

prompt1 = PromptTemplate(
    template="""
Give report on {topic}.
""",
input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="""
Give short summary of {text}.
""",
input_variables=['text']
)

parser = StrOutputParser()

chain  = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Indian cricket'})

print(result)

chain.get_graph().print_ascii()

