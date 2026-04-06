from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = WebBaseLoader(web_path='https://www.amazon.in/Lloyd-Inverter-Convertible-Anti-Viral-GLS12I5FWBEV/dp/B0BRKXFVVJ/ref=sr_1_1?_encoding=UTF8&content-id=amzn1.sym.58c90a12-100b-4a2f-8e15-7c06f1abe2be&dib=eyJ2IjoiMSJ9.82enHa5cxnCTt5ELiMlB9JpLwLyBRwApOuf02YwAWqxkPzfn76bDH_MMxIVjKxfxAERII9NElT6amrK1ikPfX7swY9r86Ikf59GimZ7CyTtnR51hYYOS4-6RHr9skXszTFOgNuQpCGWVX5Rl80ff1IucS5fn82t8edSmPMXdKB79CM72IKbQkXtVi6tcg_RYirA5hDVrSVKnM2-qr71bSUlVgGYuPvZGc6irTLUDZ0fnsGYN7U4R9IWtw8ue_mSvBB6Qddd0NXkrxNpBg_uhqQ.hdv2qAYuR7OJKAJIqFJbnj52FBWzrgb-NmSd_KVeZak&dib_tag=se&pd_rd_r=40c58606-e1d6-4638-8dd0-471791d06da1&pd_rd_w=dBm2G&pd_rd_wg=2vJcd&qid=1775503017&refinements=p_85%3A10440599031&rps=1&s=kitchen&sr=1-1')
docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

model = ChatOpenAI()

prompt = PromptTemplate(
    template='provide answer for the given question {question} from the text {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'question':'What is the product price?', 'text':docs[0].page_content})
print(result)


