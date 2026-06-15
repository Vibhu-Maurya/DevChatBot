import os
import json
import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from search_engine import search_docs
from llm_client import generate_answer

app = FastAPI(
    title="DevChatBot Search and RAG API",
    description="FastAPI service for semantic search and LLM-powered RAG documentation answering.",
    version="1.0.0"
)

# Enable CORS for future frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Telemetry logging helper
def log_api_telemetry(endpoint: str, query: str, retrieval_ms: float, llm_ms: float, total_ms: float):
    metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    log_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoint": endpoint,
        "query": query,
        "retrieval_ms": round(retrieval_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round(total_ms, 2)
    }
    try:
        with open(os.path.join(metrics_dir, "api_stats.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to log API telemetry: {e}")

class SearchRequest(BaseModel):
    query: str = Field(..., description="The query string to search for")
    top_k: int = Field(5, description="Number of results to retrieve")
    min_score: float = Field(0.60, description="Minimum similarity score threshold")
    source: Optional[str] = Field(None, description="Optional source filter (e.g., 'python', 'fastapi')")

class SearchResultItem(BaseModel):
    score: float
    title: str
    section: str
    url: str
    content: str
    source: str
    domain: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    latency_ms: float

class SourceCitation(BaseModel):
    title: str
    url: str

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    debug_prompt: Optional[str] = None

class AskRequest(BaseModel):
    query: str = Field(..., description="Question to answer using retrieved context")
    top_k: int = Field(5, description="Number of context documents to retrieve")
    min_score: float = Field(0.60, description="Minimum similarity score threshold")
    source: Optional[str] = Field(None, description="Optional source filter (e.g., 'python', 'fastapi')")
    llm_provider: Optional[str] = Field(None, description="LLM provider (gemini or openai)")
    debug: bool = Field(False, description="If true, returns the built prompt and skips the LLM call")

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    search_res = search_docs(
        query_str=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        source_filter=request.source
    )
    
    if "error" in search_res:
        raise HTTPException(status_code=500, detail=f"Database query error: {search_res['error']}")
        
    log_api_telemetry(
        endpoint="/search",
        query=request.query,
        retrieval_ms=search_res["total_ms"],
        llm_ms=0.0,
        total_ms=search_res["total_ms"]
    )
    
    return SearchResponse(
        query=request.query,
        results=search_res["results"],
        latency_ms=search_res["total_ms"]
    )

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    import time
    t_start = time.time()
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    # 1. Retrieve context chunks
    search_res = search_docs(
        query_str=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        source_filter=request.source
    )
    
    if "error" in search_res:
        raise HTTPException(status_code=500, detail=f"Database query error: {search_res['error']}")
        
    retrieval_ms = search_res["total_ms"]
    
    # 3. Deduplicate and format sources
    seen_urls = set()
    sources = []
    for item in search_res["results"]:
        url = item["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append(SourceCitation(title=item["title"], url=url))

    # If debug mode, compile the prompt and return immediately
    if request.debug:
        from llm_client import build_prompt
        compiled_prompt = build_prompt(request.query, search_res["results"])
        total_ms = (time.time() - t_start) * 1000
        return AskResponse(
            answer="Debug mode: LLM call skipped. See debug_prompt field.",
            sources=sources,
            retrieval_ms=retrieval_ms,
            llm_ms=0.0,
            total_ms=total_ms,
            debug_prompt=compiled_prompt
        )
    
    # 2. Call LLM with context
    try:
        if "debug" in request.query.lower() or "error" in request.query.lower():
            from agent_client import generate_agent_answer
            answer, llm_ms = generate_agent_answer(request.query, request.llm_provider)
            # We don't have explicit structured sources from the agent, so we pass the ones we found,
            # or an empty list if the agent bypassed them.
        else:
            answer, provider_used, llm_ms = generate_answer(
                query_str=request.query,
                contexts=search_res["results"],
                provider=request.llm_provider
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM integration error: {e}")
        
    total_ms = (time.time() - t_start) * 1000
            
    # 4. Log telemetry
    log_api_telemetry(
        endpoint="/ask",
        query=request.query,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        total_ms=total_ms
    )
    
    return AskResponse(
        answer=answer,
        sources=sources,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        total_ms=total_ms
    )

class VisionResponse(BaseModel):
    answer: str
    vision_ms: float

@app.post("/ask-vision", response_model=VisionResponse)
async def ask_vision_endpoint(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None)
):
    """
    Endpoint to analyze screenshots or images using Gemini Vision API.
    """
    import time
    from vision_client import analyze_image
    
    t_start = time.time()
    
    # Read the image bytes
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    
    # Process image with Gemini Vision Model
    answer = analyze_image(image_bytes, mime_type, query)
    
    vision_ms = (time.time() - t_start) * 1000
    
    return VisionResponse(
        answer=answer,
        vision_ms=round(vision_ms, 2)
    )

# Mount the frontend directory as StaticFiles
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
