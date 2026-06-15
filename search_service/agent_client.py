import os
import time
from typing import List, Dict, Any, Tuple
from config import settings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

from search_engine import search_docs

# Set the Tavily key explicitly in environment so the tool can pick it up
if settings.TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

@tool
def search_local_documentation(query: str, top_k: int = 5) -> str:
    """Search the local DevChatBot vector database for documentation on FastAPI, Docker, and Python. Use this first before searching the web."""
    results = search_docs(query, top_k=top_k)
    if "error" in results:
        return f"Error searching database: {results['error']}"
    
    docs = results.get("results", [])
    if not docs:
        return "No relevant documentation found locally."
        
    formatted = []
    for idx, d in enumerate(docs):
        formatted.append(f"[{idx+1}] Title: {d['title']}\nContent: {d['content']}\nURL: {d['url']}")
    return "\n\n".join(formatted)

def get_agent(provider: str = None):
    primary_provider = provider or settings.DEFAULT_LLM_PROVIDER
    
    if primary_provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=settings.DEFAULT_LLM_MODEL, google_api_key=settings.GEMINI_API_KEY, temperature=0.0)
    else:
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY, temperature=0.0)

    # Initialize tools
    tools = [search_local_documentation, PythonREPLTool()]
    
    if settings.TAVILY_API_KEY:
        tavily_tool = TavilySearchResults(max_results=3)
        tools.append(tavily_tool)

    system_prompt = "You are an expert developer assistant and debugging agent. You can search local documentation, search the web, and execute Python code to help the user. Always prioritize searching local documentation first. When providing facts, cite your sources. If you need to debug something, you can write and run python scripts to verify your answer."
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent

def generate_agent_answer(query_str: str, provider: str = None) -> Tuple[str, float]:
    """
    Generate an answer using the advanced Agent Workflow Router.
    """
    from workflows import route_query
    
    t0 = time.time()
    try:
        answer = route_query(query_str)
    except Exception as e:
        answer = f"Workflow Error: {e}"
        
    llm_ms = (time.time() - t0) * 1000
    return answer, round(llm_ms, 2)
