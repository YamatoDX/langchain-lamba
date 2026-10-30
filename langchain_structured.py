from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

class llm_schema(BaseModel):
    setup: str = Field(description="This is the main joke setup")
    punchline: str = Field(description="Main Punchline of the joke")

llm_structured = llm.with_structured_output(llm_schema)

dynamic_template = ChatPromptTemplate.from_messages([
    ("system", "answer as the berrus the destroyer God!"),
    ("user", "Tell me a {topic}")
])
ready_template = dynamic_template.invoke({"topic":"joke"})

response = llm_structured.invoke(ready_template)
print(type(response))
print(response.setup)
print(response.punchline)
