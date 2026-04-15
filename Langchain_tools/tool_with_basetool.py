from langchain_community.tools import BaseTool
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAI
from typing import Type
import os

load_dotenv()

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

class Multiply(BaseTool):
    name  : str="multiply"
    description : str="Multiplication of two numbers"
    args_schema : Type[BaseModel] = MultiplyInput

    def _run(self, a:int , b:int) -> int :
        return a*b


tool = Multiply()

result = tool.invoke({'a':5, 'b':10})
print(result)
print(tool.name)
print(tool.args)
print(tool.description)


