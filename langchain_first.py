import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
response = llm.invoke("Tell me two facts about Bangladesh!")
print(response)