import os
import requests
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

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def call_gemini_api(history: list, user_message: str):
    """Calls Gemini API directly via REST."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Format contents
    contents = []
    for msg in history:
        role = 'user' if msg['sender'] == 'user' else 'model'
        parts = []
        if msg.get('content'):
            parts.append({'text': msg['content']})
        # Check if we should add file data? 
        # For now, just context text to keep it simple unless we want Vision.
        # User prompt didn't strictly demand Vision, just Storage.
        if parts:
            contents.append({'role': role, 'parts': parts})
    
    # Add current message
    parts = []
    if user_message:
        parts.append({'text': user_message})
        
    # If no text (file only message), provide a default prompt for the AI to acknowledge the file
    if not parts:
         parts.append({'text': "[User uploaded a file]"})

    contents.append({'role': 'user', 'parts': parts})
    
    payload = {
        "contents": contents
    }
    
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        error_msg = response.text
        try:
            error_json = response.json()
            error_msg = error_json.get('error', {}).get('message', error_msg)
        except:
            pass
        raise ValueError(f"Gemini API Error ({response.status_code}): {error_msg}")
        
    data = response.json()
    try:
        return data['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError) as e:
        return "I'm sorry, I couldn't generate a response."

# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
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
        
        # Verify token matches a valid user
        get_user_from_token(token)
        
        print(f"Processing message for chat {chat_id}...")
        
        if not message and not file:
             raise HTTPException(status_code=400, detail="Message or File is required")

        user_content = message if message else ""
        file_path = None
        signed_url = None

        # 1. Handle File Upload
        if file:
            print(f"Uploading file: {file.filename}")
            file_path = upload_file_to_storage(file, chat_id)
            signed_url = get_signed_url(file_path)
            print(f"File uploaded to: {file_path}")

        # 2. Fetch Context (for AI history)
        try:
            history = fetch_context(chat_id)
        except Exception as e:
            print(f"DB Error (Context): {e}")
            raise ValueError(f"Database Error: {e}")
        
        print(f"Calling Gemini API (History: {len(history)} msgs)...")
        
        # 3. Generate Response via REST
        # Note: We are currently NOT sending the image to Gemini (Vision) to keep this step robust.
        # We are simply notifying Gemini that a file exists if message is empty.
        ai_reply = call_gemini_api(history, user_content)
        
        print("Received AI Response. Saving...")

        # 4. Save to Database using Service Key (Admin)
        # User Message
        supabase.table('messages').insert({
            'chat_id': chat_id,
            'sender': 'user',
            'content': user_content,
            'file_path': file_path
        }).execute()
        
        # AI Reply
        supabase.table('messages').insert({
            'chat_id': chat_id,
            'sender': 'ai',
            'content': ai_reply
        }).execute()
        
        print("Saved to DB.")
        return ChatResponse(reply=ai_reply, file_url=signed_url)

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
