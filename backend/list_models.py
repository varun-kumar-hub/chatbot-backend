import os
import requests
from dotenv import load_dotenv

# Load env vars
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code == 200:
    models = response.json().get('models', [])
    print("Available Models:")
    for m in models:
        name = m['name']
        if '1.5-flash' in name and 'generateContent' in m.get('supportedGenerationMethods', []):
            print(f"- {name}")
else:
    print(f"Error listing models: {response.status_code} - {response.text}")
