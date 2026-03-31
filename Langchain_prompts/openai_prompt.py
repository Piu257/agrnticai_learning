from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

topic = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

size = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

prompt_template = load_prompt('prompt_template.json')
final_prompt = prompt_template.format(paper_input=topic, style_input=style, length_input=size)
model  = ChatOpenAI(model='gpt-4')

if st.button('Summarize'):
   result = model.invoke(final_prompt)
   st.write(result)

