import requests
import os

# Test Health
try:
    print("Testing Health Check...")
    r = requests.get("http://127.0.0.1:8000/")
    print(f"Health Status: {r.status_code}")
    print(r.json())
except Exception as e:
    print(f"Health Check Failed: {e}")

# Test Chat (Will fail Auth, but should return 401, not 500 or Connection Refused)
try:
    print("\nTesting Chat Endpoint (Expect 401)...")
    r = requests.post("http://127.0.0.1:8000/chat", json={"chat_id": "123", "message": "test"})
    print(f"Chat Status: {r.status_code}")
    print(r.text)
except Exception as e:
    print(f"Chat Request Failed: {e}")
