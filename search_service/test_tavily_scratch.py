import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set explicit TAVILY_API_KEY if not picked up
from config import settings
if settings.TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

from langchain_community.tools.tavily_search import TavilySearchResults

def test_tavily():
    print(f"Using TAVILY_API_KEY starting with: {os.environ.get('TAVILY_API_KEY', '')[:10]}...")
    tool = TavilySearchResults(max_results=2)
    
    print("\nTesting search with query: 'Latest news in AI 2024'")
    try:
        results = tool.invoke("Latest news in AI 2024")
        if isinstance(results, list) and len(results) > 0:
            print("\n✅ Success! Tavily search returned results:")
            for idx, res in enumerate(results):
                print(f"\nResult {idx + 1}:")
                print(f"URL: {res.get('url')}")
                print(f"Content: {res.get('content')[:200]}...")
        else:
            print("\n⚠️ Request succeeded but no results were returned.")
            print(f"Raw output: {results}")
    except Exception as e:
        print(f"\n❌ Error calling Tavily API: {e}")

if __name__ == "__main__":
    test_tavily()
