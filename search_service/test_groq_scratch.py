import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import settings
from langchain_groq import ChatGroq

def test_groq():
    api_key = settings.GROQ_API_KEY
    if not api_key:
        print("❌ Error: GROQ_API_KEY is not set.")
        return
        
    print(f"Using GROQ_API_KEY starting with: {api_key[:10]}...")
    
    # Using llama-3.1-8b-instant as configured
    model_name = "llama-3.1-8b-instant"
    print(f"Testing Groq API with model: {model_name}")
    
    try:
        llm = ChatGroq(
            model=model_name, 
            groq_api_key=api_key, 
            temperature=0.0
        )
        
        # Test a simple prompt
        response = llm.invoke("What is 5+5? Reply in one short sentence.")
        print("\n✅ Success! Groq API returned response:")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"\n❌ Error calling Groq API: {e}")

if __name__ == "__main__":
    test_groq()
