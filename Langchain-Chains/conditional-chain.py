import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm_endpoint = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-Coder-Next',
    task='text-generation',
    huggingfacehub_api_token=api_key
)

model = ChatHuggingFace(llm= llm_endpoint)

class Sentiment(BaseModel):
    sentiment: Literal['Positive','Negative'] = Field(description='Sentiment of the feedback')


pydanticOpParser = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template="""
Determine the sentiment of the product feedback ,\n
{feedback}.
Provide the output in {format_instruction}
""",
input_variables=['feedback'],
partial_variables={'format_instruction': pydanticOpParser.get_format_instructions()}
)

classifier_chain  = prompt1 | model | pydanticOpParser

prompt2 = PromptTemplate(
    template="""
prepare a positive user reply based on the positive feedback sentiment {sentiment}.
""",
input_variables=['sentiment']
)

prompt3 =PromptTemplate(
template="""
prepare a appology user reply based on the negative feedback sentiment {sentiment}.
""",
input_variables=['sentiment']
)

parser = StrOutputParser()

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'Could not find the sentiment') 
)
chain  = classifier_chain | branch_chain 

result = chain.invoke({'feedback':'The product is terrible'})

print(result)

chain.get_graph().print_ascii()

