from langchain_community.tools import StructuredTool
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAI
import os

load_dotenv()

class Multiply(BaseModel):
    a : int = Field(description='number to multiply')
    b : int = Field(description='number to multiply')

def multiply_func(a:int , b:int) -> int :
    """Multiplication of two numbers"""
    return a*b


tool = StructuredTool.from_function(
    func=multiply_func,
    name='multiply',
    description="Multiplication of two numbers",
    args_schema=Multiply
)

result = tool.invoke({'a':5, 'b':10})
print(result)
print(tool.name)
print(tool.args)
print(tool.description)


