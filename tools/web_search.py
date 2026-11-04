import os
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
load_dotenv()

search = GoogleSerperAPIWrapper(
    serper_api_key=os.getenv("ACE_SERPER_API_KEY"),
    k=5,
    gl="in",
    hl="en"
)

@tool
def web_search(query: str) -> str:
    """
    Search Google for real-time information.
    Use this tool for current events, latest technology,
    documentation, and information not available in LLM knowledge.
    """
    return search.run(query)
