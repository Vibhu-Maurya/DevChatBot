import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import time
import argparse
import hashlib
import json
import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configuration
COLLECTION_NAME = "docs_v1"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384

# Initialize embedding engine
print("Initializing embedding engine...")
try:
    from fastembed import TextEmbedding
    class FastEmbedEngine:
        def __init__(self, model_name=MODEL_NAME):
            self.model = TextEmbedding(model_name=model_name)
        def embed_query(self, text):
            return next(self.model.embed([text])).tolist()
    engine = FastEmbedEngine()
    print("Embedding engine initialized using FastEmbed.")
except ImportError:
    from sentence_transformers import SentenceTransformer
    class STEngine:
        def __init__(self, model_name=MODEL_NAME):
            self.model = SentenceTransformer(model_name)
        def embed_query(self, text):
            return self.model.encode([text])[0].tolist()
    engine = STEngine()
    print("Embedding engine initialized using SentenceTransformers.")

def run_query(client, query_str, top_k=5, min_score=0.60):
    t0 = time.time()
    
    # 1. Embed query
    query_vector = engine.embed_query(query_str)
    t_embed = (time.time() - t0) * 1000
    
    # 2. Search
    t1 = time.time()
    try:
        # Resolve vector name if fastembed is active in the client
        try:
            vector_params = client.get_fastembed_vector_params()
            vector_name = list(vector_params.keys())[0]
        except Exception:
            vector_name = None

        if vector_name:
            search_response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                using=vector_name,
                limit=top_k
            )
        else:
            search_response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k
            )
        results = search_response.points
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return
    t_search = (time.time() - t1) * 1000
    total_time = (time.time() - t0) * 1000
    
    # 3. Output
    print(f"\nQuery: '{query_str}'")
    print(f"Latency: Embed = {t_embed:.1f}ms | Search = {t_search:.1f}ms | Total = {total_time:.1f}ms")
    print("-" * 80)
    
    valid_results = 0
    top_score = 0
    top_url = ""
    
    for idx, hit in enumerate(results):
        score = hit.score
        if score < min_score:
            continue
            
        payload = hit.payload
        if idx == 0:
            top_score = score
            top_url = payload.get("url", "")
            
        print(f"[{idx + 1}] Score: {score:.3f} | Source: {payload.get('source')} | Domain: {payload.get('domain')}")
        print(f"Title: {payload.get('title')}")
        print(f"Section: {payload.get('section')}")
        print(f"URL: {payload.get('url')}")
        print(f"Chunk Hash: {payload.get('chunk_hash')}")
        print("-" * 40)
        content = payload.get("content", "")
        # Print snippet
        snippet = content[:300].replace("\n", " ")
        print(f"Snippet: {snippet}...")
        print("=" * 80)
        valid_results += 1
        
    if valid_results == 0:
        print("No results met the minimum similarity score threshold.")
        
    # Telemetry logging
    os.makedirs("metrics", exist_ok=True)
    log_record = {
        "query": query_str,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_ms": round(t_embed, 2),
        "search_ms": round(t_search, 2),
        "top_score": round(top_score, 4),
        "results_returned": valid_results
    }
    try:
        with open(os.path.join("metrics", "retrieval_stats.jsonl"), "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to log retrieval metrics: {e}")

def main():
    parser = argparse.ArgumentParser(description="Query the Qdrant documentation collection.")
    parser.parse_arguments = parser.add_argument("--query", type=str, help="Search query string.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve.")
    parser.add_argument("--min-score", type=float, default=0.60, help="Minimum score threshold.")
    args = parser.parse_args()

    # Connect to Qdrant (try localhost:6333 first, fallback to path="qdrant_db")
    print("Connecting to Qdrant...")
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=5)
        client.get_collections()
        print("Connected to Qdrant server at localhost:6333.")
    except Exception:
        print("Could not connect to Qdrant server. Falling back to persistent local storage 'qdrant_db'.")
        client = QdrantClient(path="qdrant_db")

    # Resolve vector params config
    try:
        vector_params = client.get_fastembed_vector_params()
    except Exception:
        vector_params = VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)

    # If the collection doesn't exist (e.g. in-memory or empty server), tell the user
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Warning: Collection '{COLLECTION_NAME}' does not exist.")
        # We'll create it to avoid fatal crashes, but it will be empty
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vector_params,
        )

    if args.query:
        run_query(client, args.query, top_k=args.top_k, min_score=args.min_score)
    else:
        # Interactive loop
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end.")
        while True:
            try:
                query_str = input("\nAsk a question: > ").strip()
                if not query_str:
                    continue
                if query_str.lower() in ["exit", "quit"]:
                    break
                run_query(client, query_str, top_k=args.top_k, min_score=args.min_score)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExited interactive mode.")

if __name__ == "__main__":
    main()
