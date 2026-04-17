from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic import hub
import os
load_dotenv()

#llm
llm = ChatOpenAI()

#builtin tool : search tool to search from ducduck go website. 
search_tool = DuckDuckGoSearchRun()
#custom tool: to fetch Weather data
@tool
def get_weather_data(city:str)-> str:
    """function to get current weather information of a city"""
    return {"temp":30, "humidity":70, "weather":"clear sky"}  ##mock data

#prompt
prompt = hub.pull('hwchase17/react')


agent =create_react_agent(
    llm=llm,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)


agent_executor = AgentExecutor(
    tools=[search_tool,get_weather_data],
    agent=agent,
    verbose=True
)

result = agent_executor.invoke({'input':'What is the capital of India and what is its current population and wasther condition'})
print(result)
