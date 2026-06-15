import sys
import json
import httpx
import argparse

def test_search(query: str, source: str = None, top_k: int = 5):
    url = "http://localhost:8000/search"
    payload = {
        "query": query,
        "top_k": top_k,
        "min_score": 0.50
    }
    if source:
        payload["source"] = source
        
    print(f"Sending POST to {url} with query='{query}'...")
    try:
        response = httpx.post(url, json=payload, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse (Latency: {data['latency_ms']:.1f}ms):")
            print(f"Query: '{data['query']}'")
            print("-" * 80)
            for idx, item in enumerate(data["results"]):
                print(f"[{idx + 1}] Score: {item['score']:.4f} | Source: {item['source']} | Title: {item['title']}")
                print(f"    URL: {item['url']}")
                snippet = item['content'][:200].replace('\n', ' ')
                print(f"    Snippet: {snippet}...")
                print("-" * 40)
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Failed to connect to the FastAPI search service: {e}")
        print("Make sure the server is running with: python -m uvicorn main:app --reload")

def test_ask(query: str, source: str = None, provider: str = None, debug: bool = False):
    url = "http://localhost:8000/ask"
    payload = {
        "query": query,
        "debug": debug
    }
    if source:
        payload["source"] = source
    if provider:
        payload["llm_provider"] = provider
        
    print(f"Sending POST to {url} with query='{query}' (debug={debug})...")
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:")
            print(f"Answer: {data['answer']}")
            print("-" * 80)
            print("Sources:")
            for src in data.get("sources", []):
                print(f" - {src['title']} ({src['url']})")
            print("-" * 80)
            print(f"Latency: Retrieval = {data.get('retrieval_ms')}ms | LLM = {data.get('llm_ms')}ms | Total = {data.get('total_ms')}ms")
            if data.get("debug_prompt"):
                print("\n" + "="*40 + " DEBUG PROMPT " + "="*40)
                print(data["debug_prompt"])
                print("=" * 94)
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Failed to connect to the FastAPI search service: {e}")
        print("Make sure the server is running with: python -m uvicorn main:app --reload")

def main():
    parser = argparse.ArgumentParser(description="Test FastAPI Search Service")
    parser.add_argument("endpoint", choices=["search", "ask"], help="Endpoint to test")
    parser.add_argument("query", help="Query / Question string")
    parser.add_argument("--source", default=None, help="Filter by source (e.g. fastapi, python)")
    parser.add_argument("--provider", default=None, help="LLM provider (e.g. gemini, openai)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--debug", action="store_true", help="Debug mode (offline prompt generation check)")
    
    args = parser.parse_args()
    
    if args.endpoint == "search":
        test_search(args.query, args.source, args.top_k)
    elif args.endpoint == "ask":
        test_ask(args.query, args.source, args.provider, args.debug)

if __name__ == "__main__":
    main()

