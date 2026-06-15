import sys
from search_engine import search_docs

print("Searching database...")
res = search_docs('How do I return JSON responses in FastAPI?')

if "error" in res:
    print("Error:", res["error"])
    sys.exit(1)

print('\n--- Search Results ---')
for idx, r in enumerate(res['results']):
    print(f"[{idx+1}] Score: {r['score']}")
    print(f"Title: {r['title']}")
    print(f"URL: {r['url']}")
    snippet = r['content'][:200].replace('\n', ' ')
    print(f"Snippet: {snippet}...")
    print('-' * 40)
