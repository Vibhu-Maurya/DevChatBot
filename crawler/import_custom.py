import json
import hashlib

input_file = "../python_doc_index.json"
output_file = "chunks.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "a", encoding="utf-8") as f:
    for i, item in enumerate(data):
        content = item.get("description", "")
        if not content:
            content = item.get("title", "")
        
        full_content = f"{item.get('title')}: {content}"
        chunk_hash = hashlib.sha256(full_content.encode("utf-8")).hexdigest()
        
        url = item.get("url", "")
        if not url.startswith("http"):
            url = "https://docs.python.org/3/" + url.lstrip("/")
            
        chunk = {
            "chunk_id": f"custom_{i}",
            "chunk_hash": chunk_hash,
            "url": url,
            "domain": "docs.python.org",
            "source": "python",
            "title": item.get("title", ""),
            "section": item.get("section", ""),
            "content": full_content
        }
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"Successfully formatted and appended {len(data)} items to {output_file}")
