import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


parser = StrOutputParser()

instructions = '''
    Will return positive if the text has positive emotion,
    otherwise will return negative
'''

class llm_schema(BaseModel):
    emotion_result: Literal["positive", "negative"] = Field(description=instructions)

llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
llm_structured = llm.with_structured_output(llm_schema)

template = PromptTemplate.from_template("Generate a positive or negative emotion from the following text {user_input}")
later_template = PromptTemplate.from_template("Write 20 words about {topic} feeling")

def result_generator_function(input: llm_schema) -> str:
    return input.model_dump()["emotion_result"]

def later_generator(input: str) -> dict:
    return {"topic": input}


base_chain = template | llm_structured | RunnableLambda(result_generator_function)
postive_chain = RunnableLambda(later_generator) | later_template | llm | parser
negative_chain = RunnableLambda(later_generator) | later_template | llm | parser
default_chain = RunnableLambda(later_generator) | later_template | llm | parser

conditional = RunnableBranch(
    (lambda x: x == "positive", postive_chain),
    (lambda x: x == "negative", negative_chain),
    default_chain
)

full_chain = base_chain | conditional

response = full_chain.invoke({"user_input": "Today I feel very good!"})
print(response)
