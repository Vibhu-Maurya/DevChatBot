import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import settings
from langchain_google_genai import ChatGoogleGenerativeAI

def test_gemini():
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is not set.")
        return
        
    print(f"Using GEMINI_API_KEY starting with: {api_key[:10]}...")
    
    model_name = settings.DEFAULT_LLM_MODEL or "gemini-2.5-flash"
    print(f"Testing Gemini API with model: {model_name}")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=api_key, 
            temperature=0.0
        )
        
        # Test a simple prompt
        response = llm.invoke("What is 2+2? Reply in one short sentence.")
        print("\n✅ Success! Gemini API returned response:")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"\n❌ Error calling Gemini API: {e}")

if __name__ == "__main__":
    test_gemini()
