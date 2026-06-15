import sys
import os
import requests
import time
import argparse

API_URL = "http://localhost:8000/ask-vision"

def test_image(image_path: str, query: str):
    if not os.path.exists(image_path):
        print(f"❌ Error: Could not find image at '{image_path}'")
        return
        
    print(f"📡 Sending image to {API_URL}...")
    print(f"❓ Query: {query}")
    
    # Determine basic mime type
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, mime_type)}
        data = {"query": query}
        
        t0 = time.time()
        try:
            response = requests.post(API_URL, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            print("\n✅ === Vision API Response ===")
            print(result.get("answer", result))
            print("================================")
            print(f"⏱️ Time taken: {result.get('vision_ms', 0)/1000:.2f}s")
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Details: {e.response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the DevChatBot Vision API endpoint with a real image")
    parser.add_argument("image_path", help="Path to the image file on your computer (e.g., screenshot.png)")
    parser.add_argument("--query", "-q", 
                        default="Analyze this screenshot. Identify all GUI elements, OCR text, and any visible code or errors.", 
                        help="Question to ask about the image")
    
    args = parser.parse_args()
    test_image(args.image_path, args.query)
