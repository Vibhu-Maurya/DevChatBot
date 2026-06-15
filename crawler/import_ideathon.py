import json
import hashlib
import os
import shutil
import subprocess

print("Backing up old Python/FastAPI data...")
# 1. Backup old data
if os.path.exists("qdrant_db"):
    shutil.move("qdrant_db", "qdrant_db_backup")
if os.path.exists("chunks.jsonl"):
    shutil.move("chunks.jsonl", "chunks.jsonl.bak")
if os.path.exists("embedding_history.json"):
    shutil.move("embedding_history.json", "embedding_history.json.bak")

# 2. Create new chunks.jsonl with Ideathon data
ideathon_text = """
The Ideathon is open to everyone — developers, MBAs, designers, students, first-timers. Pick the challenge that fits you best and pitch your idea. No code required for two out of three.

Challenge 1 — Build an AI System For: Developers, Engineers, Technical Builders
Design a technical AI-native system that makes work smarter. Think autonomous agents, intelligent search, multi-model coordination, or AI copilots that actually change how people get things done.
"""

chunk = {
    "chunk_id": "ideathon_1",
    "chunk_hash": hashlib.sha256(ideathon_text.encode("utf-8")).hexdigest(),
    "url": "https://ideathon.internal/rules",
    "domain": "ideathon.internal",
    "source": "ideathon",
    "title": "Ideathon Official Rules and Challenges",
    "section": "Overview",
    "content": ideathon_text.strip()
}

with open("chunks.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("Created chunks.jsonl with Ideathon data.")

# 3. Run embed.py
print("Running embed.py to embed the Ideathon data into a fresh Qdrant DB...")
subprocess.run(["python", "embed.py"], check=True)
print("Done! The RAG system is now an Ideathon expert.")
