import os
import json
import aiohttp
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import StreamingResponse
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
# --- Helpers ---
import io
import base64
import requests
from pypdf import PdfReader

# --- Helpers ---
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF Error: {e}")
        return "[Error extracting text from PDF]"

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

def upload_file_to_storage(file: UploadFile, chat_id: str, file_content: bytes):
    """Uploads file to Supabase Storage."""
    try:
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

# System Prompt to teach AI about tools
SYSTEM_INSTRUCTION = """
You are an advanced AI assistant.
1. IMAGE GENERATION: If the user asks you to generate, create, or show an image/picture/photo, you MUST output a special tag: ((GENERATE_IMAGE: <detailed_search_query>)). 
   Example: User: "Show me a cybercity" -> You: "Here is a cybercity: ((GENERATE_IMAGE: futuristic cyberpunk city neon lights))"
   Do not try to provide a URL yourself. Use the tag.
2. FILE CONTEXT: You may receive file contents in the prompt. Use this to answer questions about the file.
"""

async def stream_gemini_api(history: list, user_message: str, image_data: dict = None):
    """Calls Gemini API via REST with streaming (Async). Supports Text + Image."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:streamGenerateContent?key={GEMINI_API_KEY}"
    
    contents = []
    
    # History (Text Only for now to save tokens/complexity)
    for msg in history:
        role = 'user' if msg['sender'] == 'user' else 'model'
        parts = []
        if msg.get('content'):
            parts.append({'text': msg['content']})
        if parts:
            contents.append({'role': role, 'parts': parts})
    
    # Current User Message
    parts = []
    
    # System Prompt
    final_message = f"{SYSTEM_INSTRUCTION}\n\n{user_message}" if user_message else SYSTEM_INSTRUCTION
    if final_message:
        parts.append({'text': final_message})
    
    # Attach Image if present
    if image_data:
        parts.append({
            "inline_data": {
                "mime_type": image_data['mime_type'],
                "data": image_data['data']
            }
        })
    elif not final_message:
         parts.append({'text': "[System: User uploaded file only]"})

    contents.append({'role': 'user', 'parts': parts})
    
    payload = {"contents": contents}
    headers = {'Content-Type': 'application/json'}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as response:
            if response.status != 200:
                text = await response.text()
                yield f"Error: {response.status} - {text}"
                return
            
            buffer = ""
            decoder = json.JSONDecoder()
            
            async for chunk in response.content.iter_any():
                if not chunk: continue
                buffer += chunk.decode('utf-8', errors='replace')
                
                while True:
                    buffer = buffer.lstrip()
                    if not buffer: break
                    
                    if buffer.startswith(('[', ',', ']')):
                        buffer = buffer[1:]
                        continue
                        
                    try:
                        obj, idx = decoder.raw_decode(buffer)
                        buffer = buffer[idx:]
                        
                        if 'error' in obj:
                             yield f" [Gemini Error: {obj['error'].get('message', 'Unknown')}] "
                             continue
                             
                        candidates = obj.get('candidates', [])
                        if not candidates: continue
                            
                        content = candidates[0].get('content')
                        if content and 'parts' in content:
                             text_chunk = content['parts'][0].get('text', '')
                             if text_chunk: yield text_chunk
                                 
                    except json.JSONDecodeError:
                        break
                    except Exception as e:
                        yield f" [Logic Error: {e}] "
                        break

# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/image")
def generate_image_proxy(query: str = Form(...)):
    """
    Generates an image using Pollinations.ai (No API Key required).
    Returns a direct URL that generates the image on-the-fly.
    """
    try:
        # 1. Enhance prompt for better results (optional but recommended)
        enhanced_query = f"{query}, high quality, detailed, 8k resolution, cinematic lighting"
        
        # 2. URL Encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote(enhanced_query)
        
        # 3. Construct URL
        # We add a random seed to ensure a new image is generated if they ask again
        import random
        seed = random.randint(1, 10000)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_query}?seed={seed}&nologo=true"
        
        return {"url": image_url, "photographer": "AI Generator"}

    except Exception as e:
        print(f"Image Gen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        image_payload = None
        
        # 1. Handle File Upload (Parse & Store)
        if file:
            try:
                # Read content once
                file_bytes = await file.read()
                file_mime = file.content_type
                
                # A. Handle Images (Pass to Vision Model)
                if file_mime.startswith('image/'):
                    print(f"DEBUG: Processing Image {file.filename} ({file_mime})")
                    # Encode base64 for Gemini
                    b64_data = base64.b64encode(file_bytes).decode('utf-8')
                    image_payload = {
                        "mime_type": file_mime,
                        "data": b64_data
                    }
                    user_content += f"\n[Attached Image: {file.filename}]"
                    
                # B. Handle Documents (RAG / Text Extraction)
                else:
                    print(f"DEBUG: Processing Document {file.filename} ({file_mime})")
                    parsed_text = ""
                    if file.filename.lower().endswith('.pdf'):
                        parsed_text = extract_text_from_pdf(file_bytes)
                        print(f"DEBUG: Extracted PDF text length: {len(parsed_text)}")
                    elif file.filename.lower().endswith(('.txt', '.md', '.csv', '.json', '.py', '.js', '.html', '.css')):
                        parsed_text = file_bytes.decode('utf-8', errors='ignore')
                        print(f"DEBUG: Extracted Text file length: {len(parsed_text)}")
                    
                    if parsed_text:
                        print("DEBUG: Appending text to prompt...")
                        user_content += f"\n\n[Attached File Content ({file.filename})]:\n{parsed_text[:30000]}" # Limit context
                    else:
                        print("DEBUG: No text extracted from file.")
                        user_content += f"\n\n[System: The user attached a file '{file.filename}' but no text could be extracted. It might be an image-only PDF or empty.]"
                    
                # C. Upload to Storage (All files)
                file_path = upload_file_to_storage(file, chat_id, file_bytes)
                
            except Exception as e:
                print(f"File Processing Error: {e}")
                user_content += "\n[Error parsing attached file]"

        # 2. Fetch Context
        history = fetch_context(chat_id)
        
        # 3. Save User Message Immediately
        supabase.table('messages').insert({
            'chat_id': chat_id,
            'sender': 'user',
            'content': message if message else f"[Sent file: {file.filename}]", 
            'file_path': file_path
        }).execute()

        # 4. Generator for Streaming & Saving AI Reply
        async def response_generator():
            full_reply = ""
            # Pass image_payload to the streamer
            async for chunk in stream_gemini_api(history, user_content, image_payload):
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
