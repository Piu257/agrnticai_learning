from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

text="""
In today’s fast-paced world, data plays a crucial role in shaping business decisions. Organizations are increasingly relying on advanced analytics to transform raw information into meaningful insights. This not only helps in understanding current performance but also enables leaders to make informed, strategic choices for the future.

A well-designed analytics platform allows users to explore complex data in a simple and intuitive manner. By integrating multiple data sources into a unified view, it provides a comprehensive understanding of organizational structures and processes. This accessibility empowers users at all levels to interact with data and derive value from it.

Data quality is another essential factor that directly impacts the reliability of insights. Accurate, consistent, and up-to-date data ensures that decisions are based on trustworthy information. Regular data quality checks and reporting mechanisms help maintain integrity and build confidence among stakeholders.

Ultimately, the goal of leveraging data and analytics is to drive better outcomes. When organizations effectively utilize these tools, they can identify opportunities, mitigate risks, and enhance overall efficiency. This leads to stronger performance and a competitive advantage in an ever-evolving market.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap = 0
)

splits = splitter.split_text(text)
print(len(splits))
print(splits[0])