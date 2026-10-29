from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

all_messages_vegeta = [
    SystemMessage(content="Answer as if you are Vegeta, prince of all saiyan!"),
    HumanMessage(content="Tell me a joke!")
]

response = llm.invoke(all_messages_vegeta)
print(response.content,"\n\n")

all_messages_vergil = [
    SystemMessage(content="Answer as if you are Vergil, son of sparda!"),
    HumanMessage(content="Tell me a joke!")
]
response_vergil = llm.invoke(all_messages_vergil)
print(response_vergil.content)
