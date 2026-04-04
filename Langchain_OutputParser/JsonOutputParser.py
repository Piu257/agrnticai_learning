import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
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

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="""
    Give birth place, religion, talent of a person {person}. 
    {format_instruction}
""",
input_variables=['person'],
partial_variables={'format_instruction': parser.get_format_instructions()}
)


chain = template1 | model | parser 

result = chain.invoke({'person':'Rabindra Nath Tagore'})

print(result)
