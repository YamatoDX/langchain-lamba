from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

from langchain.tools import tool
from langchain.agents import create_agent

from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)


duckduckgo_api = DuckDuckGoSearchAPIWrapper(
    region="wt-wt",
    safesearch="moderate",
    time=None,
    max_results=5,
)

duckduckgo_tool = DuckDuckGoSearchResults(
    api_wrapper=duckduckgo_api,
    name="web_search",
    description=(
        "Search the internet for recent information, "
        "news, current events, software documentation, "
        "technology updates, and current facts. "
        "Use when information may have changed recently. "
        "Input: a search query string. "
        "Returns search results with source links. "
        "Do not use for calculations."
    ),
    max_results=5,
    output_format="string",
)

wikipedia_api = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=3000,
)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=wikipedia_api,
    name="wikipedia_search",
    description=(
        "Search Wikipedia for historical facts, "
        "scientific concepts, biographies, definitions, "
        "inventions, and established knowledge. "
        "Use when the user asks about background "
        "information or historical topics. "
        "Input: a topic or search query string. "
        "Avoid using for breaking news."
    ),
)

@tool(parse_docstring=True)
def joke_teller() -> str:
    '''
        This tool will be used to tell a joke if needed
    '''
    response = llm.invoke("Tell me a joke like a saiyan prince vegeta!")
    return response.content

agent = create_agent(
    model=llm,
    tools=[wikipedia_tool, duckduckgo_tool, joke_teller],
    system_prompt=(
        "Try to understand the emotion of the input",
        "If the emotion is so sad or negative, tell a joke",
        "if there is no emotion, just search what is required",
        "Searching will be either from the web or from wikipedia"
    )
)

print(agent)

