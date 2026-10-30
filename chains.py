from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()
dynamic_template_one = PromptTemplate.from_template("Find 5 information about the topic {topic}")
dynamic_template_two = ChatPromptTemplate.from_messages([
    ("system","You are a social media content creator"),
    ("user", "Write a facebook post about topic:{topic} within 100 words")
])

def dictionary_maker(text:str) -> dict:
    return {"topic": text}

dictionary_maker_instance = RunnableLambda(dictionary_maker)

chain_one = dynamic_template_one | llm | parser 
chain_two = dynamic_template_two | llm | parser 

chain = chain_one | dictionary_maker_instance | chain_two 

user_input = input("Write a topic name in just one word:\n");

if user_input in [None, ""]:
    raise Exception("User Input cannot be empty!")

response = chain.invoke({"topic":user_input})

print(response)

