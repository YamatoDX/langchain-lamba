import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()

main_template = ChatPromptTemplate.from_messages([
    ("system","You are a movie summarizer"),
    ("user", "Summarize the plot of the movie: {movie_name}")
])

linkedin_template = ChatPromptTemplate.from_messages([
    ("system", "You are a Facebook post creator"),
    ("user", "Create a marketing post based on the follow movie summary: {movie_summary}")
])

job_template = ChatPromptTemplate.from_messages([
    ("system", "You are a job posting creator"),
    ("user", "Create a job posting for a perfect role of an actor based on the following movie summary: {movie_summary}")
])

def dictionary_generator(text: str) -> dict:
    return {"movie_summary": text}


main_chain = main_template | llm | parser | RunnableLambda(dictionary_generator)
linkedin_chain = linkedin_template | llm | parser 
job_chain = job_template | llm | parser 

final_chain = main_chain | RunnableParallel(branches={"linkedin": linkedin_chain, "job": job_chain})

user_input = input("Give me a movie name:\n")
if user_input in [None, ""]:
    raise Exception("Movie name cannot be empty")

print("-------------------------------------")
data = final_chain.invoke({"movie_name": user_input})["branches"]

with open("parallel_chain.txt", "w", encoding="utf-8") as f:
    f.write("****LINKEDIN POST***** -> \n")
    f.write(data["linkedin"] + "\n\n")

    f.write("****JOB POSTING****\n")
    f.write(data["job"] + "\n")

