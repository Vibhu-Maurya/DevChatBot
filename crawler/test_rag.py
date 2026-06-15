import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import json
import hashlib
import time
import uuid
import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
COLLECTION_NAME = "docs_v1"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384
CHUNKS_FILE = "chunks.jsonl"

print("Initializing FastEmbed engine...")
from fastembed import TextEmbedding
model = TextEmbedding(model_name=MODEL_NAME)
print("FastEmbed initialized.")

# Helper: Convert SHA256 string to UUID
def get_uuid_from_hash(hash_str):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_str))

def main():
    # 1. Start in-memory client
    print("\nBooting temporary in-memory Qdrant Client...")
    client = QdrantClient(":memory:")
    
    # Get fastembed vector configuration dynamically
    vector_params = client.get_fastembed_vector_params()
    vector_name = list(vector_params.keys())[0]
    
    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vector_params,
    )
    
    # 2. Load and embed chunks
    if not os.path.exists(CHUNKS_FILE):
        print(f"Error: {CHUNKS_FILE} not found. Please run the crawler first.")
        return
        
    print(f"Reading chunks from {CHUNKS_FILE}...")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    total_chunks = len(chunks)
    print(f"Ingesting {total_chunks} chunks into the in-memory vector database...")
    
    # Embed in a single batch
    texts = [c["content"] for c in chunks]
    t_embed_start = time.time()
    vectors = [vec.tolist() for vec in model.embed(texts)]
    print(f"Generated {total_chunks} embeddings in {(time.time() - t_embed_start):.2f}s.")
    
    points = []
    now_str = datetime.date.today().isoformat()
    for idx, item in enumerate(chunks):
        chunk_hash = item["chunk_hash"]
        chunk_uuid = get_uuid_from_hash(chunk_hash)
        parent_doc_hash = hashlib.sha256(item["url"].encode()).hexdigest()
        
        payload = {
            "url": item["url"],
            "title": item["title"],
            "section": item["section"],
            "content": item["content"],
            "source": item["source"],
            "domain": item["domain"],
            "chunk_hash": chunk_hash,
            "parent_doc_hash": parent_doc_hash,
            "crawl_date": now_str
        }
        
        points.append(PointStruct(
            id=chunk_uuid,
            vector={vector_name: vectors[idx]},
            payload=payload
        ))
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Ingestion complete. Starting validation queries...")
    
    # 3. Test queries
    test_queries = [
        "How do I define a path parameter?",
        "How do dependencies work?",
        "How do I return JSON responses?"
    ]
    
    for query_str in test_queries:
        print("\n" + "="*80)
        print(f"RUNNING TEST QUERY: '{query_str}'")
        print("="*80)
        
        t0 = time.time()
        query_vector = next(model.embed([query_str])).tolist()
        t_embed = (time.time() - t0) * 1000
        
        t1 = time.time()
        try:
            # Query using query_points with pre-calculated vector
            search_response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                using=vector_name,
                limit=3
            )
            # client.query_points returns a QueryResponse object.
            # Its points attribute is a list of ScoredPoint objects.
            results = search_response.points
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return
        t_search = (time.time() - t1) * 1000
        
        print(f"Latency: Embed = {t_embed:.1f}ms | Search = {t_search:.1f}ms")
        print("-" * 80)
        
        for idx, hit in enumerate(results):
            payload = hit.payload
            print(f"Rank [{idx + 1}] | Score: {hit.score:.3f}")
            print(f"  Title:   {payload.get('title')}")
            print(f"  Section: {payload.get('section')}")
            print(f"  URL:     {payload.get('url')}")
            print(f"  Snippet: {payload.get('content')[:180].replace(chr(10), ' ')}...")
            print("-" * 40)
            
if __name__ == "__main__":
    main()
