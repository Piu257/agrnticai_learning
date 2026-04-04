import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm_endpoint = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-Coder-Next',
    task='text-generation',
    huggingfacehub_api_token=api_key
)

model = ChatHuggingFace(llm= llm_endpoint)

class Person(BaseModel):
    birth_place : str = Field(description='Birth Place of the person')
    age : int = Field(description="Age of the Person")
    talent : list[str] = Field(description='the qualities the person is known for')


parser = PydanticOutputParser(pydantic_object=Person)

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
