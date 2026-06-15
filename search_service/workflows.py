import os
import platform
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from config import settings
from search_engine import search_docs
from agent_client import get_agent

def get_llm():
    if settings.DEFAULT_LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=settings.DEFAULT_LLM_MODEL, google_api_key=settings.GEMINI_API_KEY, temperature=0.0)
    else:
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY, temperature=0.0)

def get_gemini_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=settings.GEMINI_API_KEY, temperature=0.0)

def get_groq_llm():
    return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=settings.GROQ_API_KEY, temperature=0.0)

# 1. Environment Inspection Workflow
def inspect_environment(query: str) -> str:
    """Returns a string describing the current python environment and system."""
    env_details = f"OS: {platform.system()} {platform.release()}\n"
    env_details += f"Python Version: {platform.python_version()}\n"
    env_details += f"CWD: {os.getcwd()}\n"
    env_details += f"LLM Provider: {settings.DEFAULT_LLM_PROVIDER}\n"
    
    prompt = PromptTemplate.from_template(
        "You are an environment inspector. The user asked: {query}\n\nHere is the environment context:\n{env}\n\nAnswer the user based on the environment context."
    )
    chain = prompt | get_llm() | StrOutputParser()
    return "✅ [Workflow: Environment Inspection]\n" + chain.invoke({"query": query, "env": env_details})

# 2. Chaining Workflow
def chain_workflow(query: str) -> str:
    """A sequential chain: Step 1 extracts keywords -> Step 2 searches docs -> Step 3 summarizes."""
    # Step 1: Extract
    extract_prompt = PromptTemplate.from_template("Extract the 3 most important search keywords from this query. Output ONLY the keywords separated by spaces. Query: {query}")
    extractor = extract_prompt | get_llm() | StrOutputParser()
    
    # Step 2: Search (custom function wrapped as runnable)
    def do_search(keywords_str: str):
        res = search_docs(keywords_str, top_k=3)
        docs = res.get("results", [])
        if not docs:
            return "No local documentation found."
        return "\n".join([f"- {d['title']}: {d['content'][:200]}" for d in docs])
        
    search_runnable = RunnableLambda(do_search)
    
    # Step 3: Summarize
    summarize_prompt = PromptTemplate.from_template("Answer the original query using the following documentation snippets.\n\nQuery: {query}\n\nSnippets:\n{snippets}\n\nAnswer:")
    
    chain = (
        {"query": RunnablePassthrough(), "snippets": extractor | search_runnable}
        | summarize_prompt
        | get_llm()
        | StrOutputParser()
    )
    return "✅ [Workflow: Chaining]\n" + chain.invoke(query)

# 3. Parallelization Workflow
def parallel_workflow(query: str) -> str:
    """Runs multiple retrievers/tasks in parallel and synthesizes."""
    def local_search(q: str):
        res = search_docs(q, top_k=2)
        docs = res.get("results", [])
        return str([d['title'] for d in docs]) if docs else "None"
        
    def web_search(q: str):
        if settings.TAVILY_API_KEY:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tool = TavilySearchResults(max_results=2)
            try:
                return str(tool.invoke(q))
            except Exception as e:
                return f"Web search failed: {e}"
        return "Web search disabled."

    parallel_step = RunnableParallel(
        local_results=RunnableLambda(local_search),
        web_results=RunnableLambda(web_search),
        query=RunnablePassthrough()
    )
    
    synthesize_prompt = PromptTemplate.from_template(
        "Synthesize an answer for the query using both local and web results.\nQuery: {query}\nLocal: {local_results}\nWeb: {web_results}\nAnswer:"
    )
    
    chain = parallel_step | synthesize_prompt | get_llm() | StrOutputParser()
    return "✅ [Workflow: Parallelization]\n" + chain.invoke(query)

# 4. Multi-Model Coordination Workflow
def multi_model_workflow(query: str) -> str:
    """Runs query through both Gemini and Groq, then synthesizes the best answer."""
    def call_gemini(q: str):
        try:
            return get_gemini_llm().invoke(q).content
        except Exception as e:
            return f"Gemini Error: {e}"

    def call_groq(q: str):
        try:
            return get_groq_llm().invoke(q).content
        except Exception as e:
            return f"Groq Error: {e}"

    parallel_step = RunnableParallel(
        gemini_answer=RunnableLambda(call_gemini),
        groq_answer=RunnableLambda(call_groq),
        query=RunnablePassthrough()
    )
    
    synthesize_prompt = PromptTemplate.from_template(
        "You are an AI Coordinator. A user asked a complex question: '{query}'\n\n"
        "I asked two different AI models for their answer. Here is what they said:\n"
        "Model 1 (Gemini):\n{gemini_answer}\n\n"
        "Model 2 (Groq Llama 3):\n{groq_answer}\n\n"
        "Synthesize the ultimate, most accurate and comprehensive answer by combining the best parts of both. "
        "Do not mention that you are comparing models unless explicitly asked by the user.\n\n"
        "Synthesized Answer:"
    )
    
    chain = parallel_step | synthesize_prompt | get_gemini_llm() | StrOutputParser()
    return "✅ [Workflow: Multi-Model Coordination]\n" + chain.invoke(query)

# 5. Agent and Tools (Existing ReAct Agent)
def agent_tools_workflow(query: str) -> str:
    agent = get_agent()
    result = agent.invoke({"messages": [("user", query)]})
    answer_content = result["messages"][-1].content
    if isinstance(answer_content, list):
        ans = " ".join([str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in answer_content])
    else:
        ans = str(answer_content)
    return "✅ [Workflow: Agents & Tools]\n" + ans

# 6. Routing Workflow (Main Entry)
def route_query(query: str) -> str:
    """Analyzes the query and routes it to the correct workflow."""
    router_prompt = PromptTemplate.from_template(
        "You are an intelligent router. Classify the following query into exactly ONE of these categories:\n"
        "1. ENVIRONMENT: if asking about system, python, OS, or configuration.\n"
        "2. CHAIN: if the query implies a complex multi-step research or keyword extraction.\n"
        "3. PARALLEL: if the query asks to compare local knowledge vs web knowledge, or search the web.\n"
        "4. MULTI-MODEL: if the query asks for multiple perspectives, extremely complex coding architecture, or compares AI models.\n"
        "5. AGENT: for debugging, python code execution, or general questions.\n\n"
        "Query: {query}\n\nCategory (reply with just the category name):"
    )
    
    category_chain = router_prompt | get_llm() | StrOutputParser()
    category = category_chain.invoke({"query": query}).strip().upper()
    
    print(f"Routing query to category: {category}")
    
    if "ENVIRONMENT" in category:
        return inspect_environment(query)
    elif "CHAIN" in category:
        return chain_workflow(query)
    elif "PARALLEL" in category:
        return parallel_workflow(query)
    elif "MULTI-MODEL" in category or "MULTI_MODEL" in category:
        return multi_model_workflow(query)
    else:
        return agent_tools_workflow(query)
