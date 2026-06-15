import os
from dotenv import load_dotenv
import requests

load_dotenv('.env')
r = requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f'Bearer {os.getenv("GROQ_API_KEY")}'})
print([m['id'] for m in r.json().get('data', [])])
