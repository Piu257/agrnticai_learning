import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
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
Generate notes on the given topic {topic}.
""",
input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="""
prepare a quiz containg 5 question answeres based on topic {topic}.
""",
input_variables=['topic']
)

prompt3 =PromptTemplate(
template="""
Merge the notes {note} and the quiz {quiz} into a single document type writing.
""",
input_variables=['note','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "note" : prompt1 | model | parser,
    "quiz" : prompt2 | model | parser
})

chain  = parallel_chain | prompt3 | model | parser

result = chain.invoke({'topic':'Machine Learning'})

print(result)

chain.get_graph().print_ascii()

