import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

user_input = input("Provide details about your health symptoms: ")

prompt = f"Based on the following symptoms, provide a possible diagnosis and recommended next steps: {user_input}"  

response = client.responses.create(
    model="gpt-4o-mini",
    input= prompt,
    temperature=0.4,
    timeout=10
)

print("Response from OpenAI API:", response)