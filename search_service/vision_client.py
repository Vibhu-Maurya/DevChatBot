import os
import base64
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import settings

def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_image(image_bytes: bytes, mime_type: str, query: Optional[str] = None) -> str:
    """
    Sends an image and a query to Gemini's Vision model.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    llm = ChatGoogleGenerativeAI(
        model=settings.DEFAULT_LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0
    )
    
    base64_image = encode_image(image_bytes)
    image_url = f"data:{mime_type};base64,{base64_image}"
    
    prompt_text = query if query else (
        "You are an expert developer assistant with perfect vision capabilities. "
        "Please analyze this screenshot. Identify all GUI elements, perform OCR on any visible text, "
        "and explain any architecture diagrams, code, or error logs you see."
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ]
    )
    
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Vision API Error: {e}"
