from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

user_input = input("Write your topic is just one word!\n")
dynamic_prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
ready_prompt = dynamic_prompt.invoke({"topic":user_input})
response = llm.invoke(ready_prompt).content
print(response,"\n\n\n")

user_input = input("Write another topic name:\n")
dynamic_prompt = ChatPromptTemplate.from_messages([
    ("system","Answer as if you are the Beerus the destroyer God!"),
    ("user", "Tell me a joke about {topic}")
])
ready_prompt = dynamic_prompt.invoke({"topic":user_input})
response = llm.invoke(ready_prompt).content
print(response)


