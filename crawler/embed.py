import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import time
import json
import uuid
import hashlib
import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
CHUNKS_FILE = "chunks.jsonl"
HISTORY_FILE = "embedding_history.json"
COLLECTION_NAME = "docs_v1"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384
BATCH_SIZE = 128

# Set up local embedding engine (FastEmbed first, SentenceTransformers as fallback)
print("Initializing embedding engine...")
try:
    from fastembed import TextEmbedding
    class FastEmbedEngine:
        def __init__(self, model_name=MODEL_NAME):
            self.model = TextEmbedding(model_name=model_name)
        def embed(self, texts):
            # TextEmbedding.embed returns a generator of numpy arrays
            return [vec.tolist() for vec in self.model.embed(texts)]
    engine = FastEmbedEngine()
    print("Embedding engine initialized successfully using FastEmbed.")
except Exception as e:
    print(f"Failed to load FastEmbed, falling back to SentenceTransformers. Error: {e}")
    try:
        from sentence_transformers import SentenceTransformer
        class STEngine:
            def __init__(self, model_name=MODEL_NAME):
                self.model = SentenceTransformer(model_name)
            def embed(self, texts):
                return self.model.encode(texts).tolist()
        engine = STEngine()
        print("Embedding engine initialized successfully using SentenceTransformers.")
    except Exception as st_err:
        print(f"Error: SentenceTransformers fallback also failed: {st_err}")
        exit(1)

# Helper: Convert SHA256 string to UUID
def get_uuid_from_hash(hash_str):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_str))

def main():
    # 1. Connect to Qdrant (try localhost:6333 first, fallback to path="qdrant_db")
    print("Connecting to Qdrant...")
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=5)
        # Quick health check
        client.get_collections()
        print("Connected to Qdrant server at localhost:6333.")
    except Exception as conn_err:
        print("\n" + "="*80)
        print("WARNING: Could not connect to local Qdrant server at localhost:6333.")
        print("To persist embeddings, make sure Qdrant is running in Docker:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        print("Falling back to persistent local storage 'qdrant_db' for this execution.")
        print("="*80 + "\n")
        client = QdrantClient(path="qdrant_db")

    # Resolve vector name if fastembed is active in the client
    try:
        vector_params = client.get_fastembed_vector_params()
        vector_name = list(vector_params.keys())[0]
    except Exception:
        vector_params = VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
        vector_name = None

    # 2. Ensure versioned collection exists
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating Qdrant collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vector_params,
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

    # Get points count before indexing
    try:
        existing_count = client.get_collection(COLLECTION_NAME).points_count
    except Exception:
        existing_count = 0

    # 3. Load embedding history cache
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}
    else:
        history = {}

    # 4. Read chunks
    if not os.path.exists(CHUNKS_FILE):
        print(f"Error: {CHUNKS_FILE} not found. Please run the crawler first.")
        return

    # Start timing embedding execution
    t_start = time.time()

    print(f"Reading chunks from {CHUNKS_FILE}...")
    new_chunks = []
    chunks_processed = 0
    chunks_skipped = 0
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_hash = chunk.get("chunk_hash")
            chunks_processed += 1
            
            # Check cache
            cached_info = history.get(chunk_hash)
            if cached_info and cached_info.get("model") == MODEL_NAME:
                chunks_skipped += 1
                continue # Skip
            
            new_chunks.append(chunk)

    total_new = len(new_chunks)
    chunks_embedded = total_new
    print(f"Found {total_new} new or updated chunks to embed.")

    if total_new > 0:
        # 5. Batch process and upsert
        processed_count = 0
        now_str = datetime.date.today().isoformat()

        for i in range(0, total_new, BATCH_SIZE):
            batch = new_chunks[i : i + BATCH_SIZE]
            texts = [item["content"] for item in batch]
            
            print(f"Processing batch of size {len(batch)} ({processed_count + len(batch)}/{total_new})...")
            vectors = engine.embed(texts)
            
            points = []
            for idx, item in enumerate(batch):
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
                
                vec_data = {vector_name: vectors[idx]} if vector_name else vectors[idx]
                
                points.append(PointStruct(
                    id=chunk_uuid,
                    vector=vec_data,
                    payload=payload
                ))

            # Bulk upsert to Qdrant
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            
            # Update history cache keys
            for item in batch:
                history[item["chunk_hash"]] = {
                    "embedded_at": now_str,
                    "model": MODEL_NAME
                }
                
            processed_count += len(batch)

        # 6. Save history cache atomically
        print("Saving embedding history...")
        tmp_history = HISTORY_FILE + ".tmp"
        try:
            with open(tmp_history, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            os.replace(tmp_history, HISTORY_FILE)
            print("Embedding history saved.")
        except Exception as e:
            print(f"Error saving embedding history: {e}")

    # Calculate metrics
    t_end = time.time()
    embedding_time_seconds = round(t_end - t_start, 2)
    chunks_per_second = round(chunks_embedded / embedding_time_seconds, 2) if embedding_time_seconds > 0 else 0.0

    # Ensure metrics directory exists
    os.makedirs("metrics", exist_ok=True)
    
    embed_stats = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "chunks_processed": chunks_processed,
        "chunks_embedded": chunks_embedded,
        "chunks_skipped": chunks_skipped,
        "embedding_time_seconds": embedding_time_seconds,
        "model": MODEL_NAME,
        "chunks_per_second": chunks_per_second
    }

    try:
        with open(os.path.join("metrics", "embed_stats.jsonl"), "a", encoding="utf-8") as sf:
            sf.write(json.dumps(embed_stats, ensure_ascii=False) + "\n")
        print("Embedding statistics logged to metrics/embed_stats.jsonl.")
    except Exception as e:
        print(f"Error saving embedding stats: {e}")

    # 7. Verification print
    try:
        col_info = client.get_collection(COLLECTION_NAME)
        print(f"\nCollection: {COLLECTION_NAME}")
        print(f"Existing points: {existing_count}")
        print(f"New points: {chunks_embedded}")
        print(f"Final points: {col_info.points_count}")
        print(f"Embedding History Count: {len(history)}")
    except Exception as e:
        print(f"Completed upsert, but failed to fetch collection count: {e}")

if __name__ == "__main__":
    main()
