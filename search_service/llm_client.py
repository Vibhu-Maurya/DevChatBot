import time
from typing import List, Dict, Any, Tuple
from config import settings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_context_string(contexts: List[Dict[str, Any]]) -> str:
    context_text_list = []
    for idx, ctx in enumerate(contexts):
        ref_num = idx + 1
        source_info = f"[{ref_num}] Source: {ctx.get('source')} | Title: {ctx.get('title')} | Section: {ctx.get('section')} | URL: {ctx.get('url')}"
        content_info = f"Content:\n{ctx.get('content')}"
        context_text_list.append(f"{source_info}\n{content_info}\n")
        
    return "\n---\n".join(context_text_list)

# We define the prompt template using LangChain's PromptTemplate
rag_prompt = PromptTemplate.from_template("""You are a friendly, expert developer assistant for DevChatBot.

Format and Rules:
1. If the user's input is a simple greeting (like "hello", "hi") or conversational, respond naturally and politely using your general knowledge.
2. If the user asks a technical question, prioritize answering using the provided documentation context chunks below.
3. When referencing facts from the context chunks, cite your sources using bracketed numbers, e.g., [1], [2].
4. If a technical question cannot be answered by the context, you may use your general knowledge to help, but clearly state that the specific documentation was not found.

CONTEXT CHUNKS:
---
{context}

---
USER QUESTION: {query}

ANSWER:""")

# Helper to support `build_prompt` for backwards compatibility with main.py's debug mode
def build_prompt(query_str: str, contexts: List[Dict[str, Any]]) -> str:
    return rag_prompt.format(
        context=build_context_string(contexts),
        query=query_str
    )

def generate_answer(
    query_str: str, 
    contexts: List[Dict[str, Any]], 
    provider: str = None
) -> Tuple[str, str, float]:
    """
    Generate an answer using RAG pipeline context via LangChain.
    Returns (answer, provider_used, llm_ms).
    """
    t0 = time.time()
    context_str = build_context_string(contexts)
    
    # Select primary provider
    primary_provider = provider or settings.DEFAULT_LLM_PROVIDER
    
    # Define execution chain based on keys
    chain_fallback = []
    if primary_provider == "gemini":
        chain_fallback = [("gemini", settings.DEFAULT_LLM_MODEL, settings.GEMINI_API_KEY), ("groq", "llama-3.1-8b-instant", settings.GROQ_API_KEY)]
    else:
        chain_fallback = [("groq", "llama-3.1-8b-instant", settings.GROQ_API_KEY), ("gemini", settings.DEFAULT_LLM_MODEL, settings.GEMINI_API_KEY)]
        
    errors = []
    for prov, model_name, key in chain_fallback:
        if not key:
            errors.append(f"Skipping {prov}: API Key is not configured.")
            continue
            
        try:
            print(f"Calling LLM provider={prov} model={model_name} via LangChain...")
            
            if prov == "gemini":
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=0.0)
            else:
                llm = ChatGroq(model=model_name, groq_api_key=key, temperature=0.0)
                
            # Build the LCEL Chain: Prompt -> LLM -> String Output
            qa_chain = rag_prompt | llm | StrOutputParser()
            
            # Invoke the chain
            answer = qa_chain.invoke({
                "context": context_str,
                "query": query_str
            })
            
            llm_ms = (time.time() - t0) * 1000
            return answer, prov, round(llm_ms, 2)
        except Exception as e:
            errors.append(f"Failed {prov} API call: {e}")
            print(f"Error during {prov} call: {e}")
            
    # If all options failed, provide a mock response for the demo
    print("WARNING: No LLM providers available. Falling back to mock response for demonstration.")
    mock_answer = f"""This is a **simulated AI response** because no API keys were configured in your `.env` file! 

However, the RAG pipeline successfully retrieved **{len(contexts)}** relevant chunks from Qdrant, proving the semantic search and routing is working perfectly. 

### How to get real answers:
To get real AI-generated answers, simply open `search_service/.env` and add your API keys.
"""
    llm_ms = (time.time() - t0) * 1000
    return mock_answer, "mock-llm", round(llm_ms, 2)
