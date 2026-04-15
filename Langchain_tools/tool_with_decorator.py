from langchain_community.tools import tool
from dotenv import load_dotenv
from langchain_openai import OpenAI
import os

load_dotenv()

@tool 
def multiply(a:int , b:int) -> int :
    """Multiply two numbers"""
    return a*b

result = multiply.invoke({'a':5,'b':10})

print(result)
print(multiply.name)
print(multiply.args)
print(multiply.description)
