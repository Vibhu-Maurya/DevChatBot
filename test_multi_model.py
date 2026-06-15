import requests
import json
import time

def test_multi_model():
    url = "http://localhost:8000/ask"
    # We trigger the MULTI-MODEL router condition by explicitly asking for it
    query = "Compare the coding architecture of a microservices backend vs a monolithic backend. Use MULTI-MODEL coordination."
    
    print(f"📡 Sending query to {url}: '{query}'")
    
    t_start = time.time()
    try:
        response = requests.post(url, json={"query": query})
        response.raise_for_status()
        data = response.json()
        
        print("\n✅ === Response ===")
        print(data.get("answer"))
        print("\n================================")
        print(f"⏱️ Time taken: {time.time() - t_start:.2f}s")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server. Is FastAPI running on port 8000?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_multi_model()
