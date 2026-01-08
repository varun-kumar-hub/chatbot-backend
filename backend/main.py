import os
import json
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'), override=True)

# --- Configuration ---
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing environment variables. Check your .env file.")

print(f"DEBUG: Service Key Loaded. Ends in: ...{SUPABASE_SERVICE_ROLE_KEY[-10:] if SUPABASE_SERVICE_ROLE_KEY else 'None'}")
print(f"DEBUG: Gemini Key Loaded. Ends in: ...{GEMINI_API_KEY[-10:] if GEMINI_API_KEY else 'None'}")

# Initialize Supabase with Service Key (Admin Access)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(redirect_slashes=False)

# Enable CORS
# IMPORTANT: Do not use "*" with allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chatbot139.vercel.app", 
        "https://chatbot-frontend-varun-kumar-hubs-projects.vercel.app", # Potential preview URL
        "http://localhost:5173", 
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (Helpers match)

# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/chat")
def chat_debug():
    """Debug endpoint to verify URL reachability."""
    return {"status": "error", "message": "You must use POST to chat. Method Not Allowed (GET). Url is correct though."}

@app.post("/chat")
async def chat_endpoint(
    chat_id: str = Form(...),
    message: str = Form(None), # Optional
    file: UploadFile = File(None),
    authorization: str = Header(None)
):
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        
        token = authorization.split(" ")[1]
        get_user_from_token(token)
        
        if not message and not file:
             raise HTTPException(status_code=400, detail="Message or File is required")

        user_content = message if message else ""
        file_path = None
        
        # 1. Handle File Upload
        if file:
            file_path = upload_file_to_storage(file, chat_id)

        # 2. Fetch Context
        history = fetch_context(chat_id)
        
        # 3. Save User Message Immediately
        supabase.table('messages').insert({
            'chat_id': chat_id,
            'sender': 'user',
            'content': user_content,
            'file_path': file_path
        }).execute()

        # 4. Generator for Streaming & Saving AI Reply
        async def response_generator():
            full_reply = ""
            # Iterate aiohttp generator asynchronously
            async for chunk in stream_gemini_api(history, user_content):
                full_reply += chunk
                yield chunk
            
            # Save AI Reply after stream finishes
            if full_reply:
                supabase.table('messages').insert({
                    'chat_id': chat_id,
                    'sender': 'ai',
                    'content': full_reply
                }).execute()
        
        return StreamingResponse(response_generator(), media_type="text/plain")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
