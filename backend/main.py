import os
import json
import aiohttp
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

# --- Models ---
class ChatResponse(BaseModel):
    reply: str
    file_url: Optional[str] = None
    
# --- Helpers ---
def get_user_from_token(token: str):
    """
    Validates the Supabase JWT and returns the User ID.
    Even though we use Service Key for DB, we must verify user identity for security.
    """
    try:
        user = supabase.auth.get_user(token)
        if not user or not user.user:
           raise HTTPException(status_code=401, detail="Invalid token (User not found)")
        return user.user.id
    except Exception as e:
        print(f"Auth Verification Failed: {e}")
        raise HTTPException(status_code=401, detail=f"Auth Failed: {str(e)}")

def get_signed_url(file_path: str):
    """Generates a signed URL for a private file (valid for 1 hour)."""
    try:
        # 3600 seconds = 1 hour
        return supabase.storage.from_("chat-files").create_signed_url(file_path, 3600)['signedURL']
    except Exception as e:
        print(f"Error generating signed URL: {e}")
        return None

def fetch_context(chat_id: str, limit: int = 15):
    """Fetches context using Service Key (Admin) client."""
    response = supabase.table('messages')\
        .select('*')\
        .eq('chat_id', chat_id)\
        .order('created_at', desc=True)\
        .limit(limit)\
        .execute()
    
    data = response.data[::-1] if response.data else []
    
    # Enrich messages with signed URLs if they have files
    for msg in data:
        if msg.get('file_path'):
            msg['file_url'] = get_signed_url(msg['file_path'])
            
    return data

def upload_file_to_storage(file: UploadFile, chat_id: str):
    """Uploads file to Supabase Storage."""
    try:
        file_content = file.file.read()
        # Sanitize filename or just use it (assuming backend validation isn't strict requirement for this demo)
        file_path = f"{chat_id}/{file.filename}"
        
        # Upload using Service Key (Admin)
        supabase.storage.from_("chat-files").upload(
            file_path,
            file_content,
            {"content-type": file.content_type, "upsert": "true"} 
        )
        return file_path
    except Exception as e:
        print(f"Storage Upload Failed: {e}")
        raise ValueError(f"File Upload Failed: {e}")

from fastapi.responses import StreamingResponse

async def stream_gemini_api(history: list, user_message: str):
    """Calls Gemini API via REST with streaming (Async)."""
    # Fix: Use stable model gemini-1.5-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GEMINI_API_KEY}"
    
    contents = []
    for msg in history:
        role = 'user' if msg['sender'] == 'user' else 'model'
        parts = []
        if msg.get('content'):
            parts.append({'text': msg['content']})
        if parts:
            contents.append({'role': role, 'parts': parts})
    
    parts = []
    if user_message:
        parts.append({'text': user_message})
    if not parts:
         parts.append({'text': "[User uploaded a file]"})

    contents.append({'role': 'user', 'parts': parts})
    
    payload = {"contents": contents}
    headers = {'Content-Type': 'application/json'}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as response:
            if response.status != 200:
                text = await response.text()
                yield f"Error: {response.status} - {text}"
                return
            
            async for line in response.content:
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line: continue
                    
                    if decoded_line.startswith(',') or decoded_line == '[' or decoded_line == ']':
                        continue
                    
                    try:
                        obj = json.loads(decoded_line)
                        if 'error' in obj:
                             yield f" [Gemini Error: {obj['error'].get('message', 'Unknown')}] "
                             continue
                             
                        candidates = obj.get('candidates', [])
                        if not candidates:
                            # Handle safety blocks or empty candidates
                            if obj.get('promptFeedback'):
                                yield " [Safety Block] "
                            continue
                            
                        content = candidates[0].get('content')
                        if content and 'parts' in content:
                             text_chunk = content['parts'][0].get('text', '')
                             if text_chunk:
                                 yield text_chunk
                    except Exception as e:
                        # For debugging purposes, yield the error briefly or log it
                        # yield f" [Parse Error: {e}] " 
                        pass

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
