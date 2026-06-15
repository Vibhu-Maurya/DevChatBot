import os
import time
from typing import List, Dict, Any, Optional

# Disable huggingface hub symlinks to prevent warning logs
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from config import settings

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384

engine = None

def get_engine():
    global engine
    if engine is None:
        print("Initializing embedding engine dynamically...")
        try:
            from fastembed import TextEmbedding
            class FastEmbedEngine:
                def __init__(self, model_name=MODEL_NAME):
                    self.model = TextEmbedding(model_name=model_name)
                def embed_query(self, text: str) -> List[float]:
                    return next(self.model.embed([text])).tolist()
            engine = FastEmbedEngine()
            print("Embedding engine initialized using FastEmbed.")
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer
                class STEngine:
                    def __init__(self, model_name=MODEL_NAME):
                        self.model = SentenceTransformer(model_name)
                    def embed_query(self, text: str) -> List[float]:
                        return self.model.encode([text])[0].tolist()
                engine = STEngine()
                print("Embedding engine initialized using SentenceTransformers.")
            except Exception as e:
                print(f"Error: Failed to initialize any embedding engine: {e}")
                raise e
    return engine

client = None

def get_client():
    global client
    if client is None:
        print("Connecting to Qdrant Client dynamically...")
        try:
            if settings.QDRANT_URL and settings.QDRANT_API_KEY:
                client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=10)
                print(f"Connected to Qdrant Cloud at {settings.QDRANT_URL}.")
            else:
                client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5)
                print(f"Connected to Qdrant server at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}.")
            client.get_collections()
        except Exception:
            # Resolve relative path for fallback db
            fallback_path = settings.QDRANT_PATH
            if not os.path.isabs(fallback_path):
                # Resolve relative to this file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                fallback_path = os.path.normpath(os.path.join(current_dir, fallback_path))
            
            print(f"Could not connect to Qdrant server. Falling back to local storage path: '{fallback_path}'.")
            client = QdrantClient(path=fallback_path)
    return client

def search_docs(
    query_str: str, 
    top_k: int = 5, 
    min_score: float = 0.60,
    source_filter: Optional[str] = None
) -> Dict[str, Any]:
    t0 = time.time()
    
    # 1. Embed query
    current_engine = get_engine()
    query_vector = current_engine.embed_query(query_str)
    t_embed = (time.time() - t0) * 1000
    
    # 2. Setup filtering if provided
    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter)
                )
            ]
        )
        
    # 3. Query Qdrant
    t1 = time.time()
    try:
        current_client = get_client()
        # Resolve vector name if fastembed is active in the client
        try:
            vector_params = current_client.get_fastembed_vector_params()
            vector_name = list(vector_params.keys())[0]
        except Exception:
            vector_name = None

        if vector_name:
            search_response = current_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_vector,
                using=vector_name,
                query_filter=query_filter,
                limit=top_k
            )
        else:
            search_response = current_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
        results = search_response.points
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return {
            "results": [],
            "embed_ms": round(t_embed, 2),
            "search_ms": round((time.time() - t1) * 1000, 2),
            "error": str(e)
        }
        
    t_search = (time.time() - t1) * 1000
    
    # 4. Process results
    hits = []
    for hit in results:
        score = hit.score
        if score < min_score:
            continue
            
        payload = hit.payload or {}
        hits.append({
            "score": round(score, 4),
            "title": payload.get("title", "Untitled"),
            "section": payload.get("section", ""),
            "url": payload.get("url", ""),
            "content": payload.get("content", ""),
            "source": payload.get("source", ""),
            "domain": payload.get("domain", "")
        })
        
    return {
        "results": hits,
        "embed_ms": round(t_embed, 2),
        "search_ms": round(t_search, 2),
        "total_ms": round((time.time() - t0) * 1000, 2)
    }
