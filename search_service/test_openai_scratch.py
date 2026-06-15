import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import settings
from langchain_openai import ChatOpenAI

def test_openai():
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        print("❌ Error: OPENAI_API_KEY is not set.")
        return
        
    print(f"Using OPENAI_API_KEY starting with: {api_key[:10]}...")
    
    # Using gpt-4o-mini as configured in agent_client.py
    model_name = "gpt-4o-mini"
    print(f"Testing OpenAI API with model: {model_name}")
    
    try:
        llm = ChatOpenAI(
            model=model_name, 
            openai_api_key=api_key, 
            temperature=0.0
        )
        
        # Test a simple prompt
        response = llm.invoke("What is 3+3? Reply in one short sentence.")
        print("\n✅ Success! OpenAI API returned response:")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"\n❌ Error calling OpenAI API: {e}")

if __name__ == "__main__":
    test_openai()
