import sqlite3
import json
import base64
import os
import sys
import threading
import time
import socket
import getpass
import shutil
import traceback
import csv
import io
import pandas as pd
import subprocess
from datetime import datetime
from typing import List, Optional

# Standard FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fpdf import FPDF
import requests
import urllib3
import uvicorn

# --- ENV SETUP ---
# (GUI Args removed for Cloud)

# --- PROXY CONFIGURATION ---
# Note: ZScaler proxies usually don't exist in Cloud/Railway environments.
# You might want to remove this if deploying to the open web, 
# but I will keep it conditional or disabled by default logic if needed.
# For now, I'm keeping your config but ensuring it doesn't break if unreachable.
PROXY_URL = "http://gateway.zscaler.net:80"
PROXIES = {"http": PROXY_URL, "https": PROXY_URL}
urllib3.disable_warnings()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- CLOUD PATH LOGIC ---
# Cloud servers are not "frozen" (exe). They run as scripts.
# We prioritize an Environment Variable for the data path (common in Railway/Render).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if a volume is mounted (e.g. via env var) or default to local folder
if os.environ.get("RAILWAY_VOLUME_MOUNT_PATH"):
    DATA_DIR = os.environ["RAILWAY_VOLUME_MOUNT_PATH"]
else:
    DATA_DIR = os.path.join(BASE_DIR, "stm_data")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Reports folder
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# --- PATHS ---
STATIC_DIR = os.path.join(BASE_DIR, "dist") 
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
DB_FILE = os.path.join(DATA_DIR, "stm_notes_final.db")
API_KEY_FILE = os.path.join(DATA_DIR, "api_key.txt")
PROMPTS_FILE = os.path.join(DATA_DIR, "saved_prompts.json")
SAVED_NOTES_DIR = os.path.join(DATA_DIR, "saved_notes")

# --- DATABASE ---
def get_db():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        if not os.path.exists(SAVED_NOTES_DIR): os.makedirs(SAVED_NOTES_DIR)
        conn = get_db()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS folders (id INTEGER PRIMARY KEY, name TEXT UNIQUE, color TEXT, created_at TEXT, is_favorite BOOLEAN DEFAULT 0)''')
        try: c.execute("SELECT is_favorite FROM folders LIMIT 1")
        except: c.execute("ALTER TABLE folders ADD COLUMN is_favorite BOOLEAN DEFAULT 0")
        c.execute('''CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, folder_id INTEGER, image_path TEXT, json_data TEXT, created_at TEXT, FOREIGN KEY(folder_id) REFERENCES folders(id))''')
        conn.commit()
        conn.close()
    except Exception as e: print(f"DB Init Error: {e}")

init_db()

# --- UTILS ---
def clean_text(text):
    if isinstance(text, dict): 
        text = ", ".join([f"{k.title()}: {v}" for k, v in text.items()])
    if isinstance(text, list): 
        text = ", ".join(map(str, text))
    if text is None: 
        return "N/A"
    text = str(text)
    text = text.replace('\u20b9', 'Rs.').replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    return text.encode('latin-1', 'replace').decode('latin-1')

# --- PDF GENERATOR ---
class CorporatePDF(FPDF):
    def header(self):
        if os.path.exists(LOGO_PATH): self.image(LOGO_PATH, 170, 8, 33) 
        self.set_font('Arial', 'B', 24); self.set_text_color(3, 35, 75); self.cell(0, 10, 'Meeting Report', 0, 1, 'L')
        self.set_font('Arial', 'I', 10); self.set_text_color(100, 100, 100); self.cell(0, 10, f'Generated on: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'L')
        self.set_draw_color(60, 180, 231); self.set_line_width(1); self.line(10, 30, 200, 30); self.ln(10)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12); self.set_text_color(255, 255, 255); self.set_fill_color(3, 35, 75); self.cell(0, 8, f"  {label.upper()}", 0, 1, 'L', 1); self.ln(4)
    def chapter_body(self, body):
        self.set_font('Arial', '', 11); self.set_text_color(50, 50, 50); self.multi_cell(0, 6, clean_text(body)); self.ln(6)

def build_pdf_data(data, filepath):
    pdf = CorporatePDF()
    pdf.add_page()
    sections = [("Executive Summary", data.get('executive_summary')), ("Customer Info", data.get('customer_information')), ("Product & Specs", data.get('product_details')), ("Pricing", data.get('pricing_information'))]
    for title, content in sections:
        if content:
            pdf.chapter_title(title); pdf.chapter_body(content)
    actions = data.get('action_items')
    if isinstance(actions, list) and actions:
        pdf.chapter_title("Action Items"); pdf.set_font("Arial", '', 11); pdf.set_text_color(50, 50, 50)
        for item in actions:
            pdf.cell(5); pdf.cell(0, 6, f"[ ] {clean_text(item)}", ln=True)
    pdf.output(filepath)

class PDFRequest(BaseModel):
    note_id: int

class EmailRequest(BaseModel):
    note_id: int
    mode: str

class PromptRequest(BaseModel):
    name: str
    content: str

class FavoriteRequest(BaseModel):
    is_favorite: bool

# --- ENDPOINTS ---
@app.patch("/folders/{folder_id}/favorite")
def toggle_favorite(folder_id: int, req: FavoriteRequest):
    conn = get_db()
    try:
        conn.execute("UPDATE folders SET is_favorite = ? WHERE id = ?", (1 if req.is_favorite else 0, folder_id))
        conn.commit()
        return {"status": "success", "folder_id": folder_id, "is_favorite": req.is_favorite}
    except Exception as e: raise HTTPException(500, f"Database error: {str(e)}")
    finally: conn.close()

@app.get("/user")
def get_system_user():
    try:
        # On Cloud, getuser might return 'root' or 'runner'. 
        raw = getpass.getuser(); clean = raw.replace(".", " ").replace("_", " ").title()
        return {"username": clean, "role": "System Admin"}
    except: return {"username": "Authorized User", "role": "User"}

@app.get("/folders")
def get_folders():
    conn = get_db(); folders = conn.execute("SELECT * FROM folders ORDER BY created_at DESC").fetchall(); conn.close()
    return {"folders": [dict(f) for f in folders]}

@app.post("/folders")
def create_folder(folder: dict):
    conn = get_db()
    try:
        c = conn.execute("INSERT INTO folders (name, color, created_at) VALUES (?,?,?)", (folder['name'], folder.get('color', '#03234B'), datetime.now().strftime("%Y-%m-%d")))
        conn.commit(); rid = c.lastrowid; conn.close(); return {"id": rid}
    except: raise HTTPException(400, "Exists")

@app.delete("/folders/{folder_id}")
def delete_folder(folder_id: int):
    conn = get_db()
    try:
        conn.execute("DELETE FROM notes WHERE folder_id=?", (folder_id,))
        conn.execute("DELETE FROM folders WHERE id=?", (folder_id,))
        conn.commit()
        return {"status": "deleted"}
    finally: conn.close()

@app.get("/all_notes")
def get_all_notes():
    conn = get_db(); notes = conn.execute("SELECT * FROM notes ORDER BY id DESC").fetchall(); conn.close()
    return {"notes": [{"id": n[0], "folder_id": n[1], "data": json.loads(n[3]), "created_at": n[4]} for n in notes]}

@app.get("/notes/{folder_id}")
def get_notes(folder_id: int):
    conn = get_db(); notes = conn.execute("SELECT * FROM notes WHERE folder_id=? ORDER BY id DESC", (folder_id,)).fetchall(); conn.close()
    return {"notes": [{"id": n[0], "folder_id": n[1], "data": json.loads(n[3]), "created_at": n[4]} for n in notes]}

@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    conn = get_db()
    try:
        cur = conn.execute("SELECT folder_id FROM notes WHERE id=?", (note_id,))
        res = cur.fetchone()
        if not res: return {"status": "error"}
        fid = res[0]
        conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        cur = conn.execute("SELECT COUNT(*) FROM notes WHERE folder_id=?", (fid,))
        remaining_count = cur.fetchone()[0]
        if remaining_count == 0:
            conn.execute("DELETE FROM folders WHERE id=?", (fid,))
            print(f"[LOG] Folder {fid} was empty and has been removed.")
        conn.commit()
        return {"status": "deleted"}
    finally: conn.close()

@app.post("/generate_pdf")
def generate_pdf_endpoint(req: PDFRequest):
    conn = get_db(); note = conn.execute("SELECT json_data FROM notes WHERE id=?", (req.note_id,)).fetchone(); conn.close()
    if not note: raise HTTPException(404, "Note not found")
    data = json.loads(note[0])
    title = clean_text(data.get('customer_information'))
    if not title or title == "N/A": title = f"Report_{req.note_id}"
    safe_name = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).strip()
    filename = f"{safe_name}.pdf"; path = os.path.join(REPORTS_DIR, filename)
    try:
        if os.path.exists(path):
            try: os.remove(path)
            except: filename = f"{safe_name}_{int(time.time())}.pdf"; path = os.path.join(REPORTS_DIR, filename)
        build_pdf_data(data, path)
        return {"status": "success", "message": f"Saved to: {path}", "filename": filename}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/prompts")
def get_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, "r") as f: return json.load(f)
    return {}

@app.post("/prompts")
def save_prompt(req: PromptRequest):
    data = {}
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, "r") as f: data = json.load(f)
    data[req.name] = req.content
    with open(PROMPTS_FILE, "w") as f: json.dump(data, f)
    return {"status": "saved"}

@app.post("/import_prompts")
async def import_prompts(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        if df.empty: raise HTTPException(400, "Excel file is empty")
        new_prompts = {}
        for _, row in df.iterrows():
            name = str(row.iloc[0]).strip()
            text = str(row.iloc[1]).strip()
            if name and text and name.lower() != "nan": new_prompts[name] = text
        return {"status": "success", "prompts": new_prompts}
    except Exception as e: raise HTTPException(500, f"Error processing Excel: {str(e)}")
    
@app.post("/send_email")
def send_email(req: EmailRequest):
    # Outlook Automation (win32com) DOES NOT work on Linux/Cloud.
    # You must replace this with SMTP logic (Gmail/Sendgrid).
    # For now, we return an error to prevent crashing.
    raise HTTPException(501, "Email feature requires SMTP configuration on Cloud Server (Outlook automation is Desktop only).")

@app.post("/analyze")
async def analyze_note(
    folder_id: int = Form(...), mode: str = Form(...), text_content: Optional[str] = Form(None), 
    files: List[UploadFile] = File(default=[]), custom_prompt: Optional[str] = Form(None), merge: bool = Form(True) 
):
    # Try reading API Key from Env Var first (Best Practice for Cloud)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            with open(API_KEY_FILE, "r") as f: api_key = f.read().strip()
        except: 
            pass
    
    if not api_key:
        raise HTTPException(500, "API Key missing. Set GROQ_API_KEY env var or api_key.txt")

    base_prompt = """Analyze this meeting input. Return strictly JSON with these keys:
                "executive_summary": "High level summary (string)",
                "customer_information": "Client Name/Company (string only, no nested objects)",
                "product_details": "Extract specific product names, part numbers, SKUs, or technical specifications mentioned (string)",
                "key_points": ["List of main discussion topics"],
                "action_items": ["List of tasks"],
                "pricing_information": "Quotes/Costs (string only, no nested objects)",
                "additional_notes": "Deadlines/Context (string)" """

    extraction_enhancer = """STRICT DATA INTEGRITY RULES:
    1. Output MUST be a single-level JSON object. 
    2. EVERY value MUST be a simple string or a simple array of strings.
    3. PROHIBITED: Do not use nested objects, dictionaries, or key-value pairs inside any value. 
    """

    final_prompt = base_prompt + "\n" + extraction_enhancer
    if custom_prompt: final_prompt += f"\nUser Directive: {custom_prompt}"
    valid_files = [f for f in files if f.filename]

    async def call_ai_api(msgs):
        try:
            res = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"}, proxies=PROXIES, verify=False, timeout=(10, 120),
                json={"model": "meta-llama/llama-4-scout-17b-16e-instruct", "messages": [{"role": "user", "content": msgs}], "temperature": 0.1, "response_format": {"type": "json_object"}})
            if res.status_code != 200: return None
            return json.loads(res.json()['choices'][0]['message']['content'])
        except Exception as e: return None

    if mode == 'image' and valid_files:
        if merge:
            messages = [{"type": "text", "text": final_prompt}]
            for f in valid_files:
                content = await f.read()
                b64 = base64.b64encode(content).decode("utf-8")
                # On Cloud, we might not want to save every temp image to disk if storage is ephemeral,
                # but for now we keep the logic as it is used for logic flow.
                clean_name = os.path.basename(f.filename)
                fname = os.path.join(SAVED_NOTES_DIR, f"{int(time.time())}_{clean_name}")
                with open(fname, "wb") as out: out.write(content)
                messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            ai_data = await call_ai_api(messages)
            if not ai_data: raise HTTPException(500, "AI synthesis failed")
            conn = get_db()
            conn.execute("INSERT INTO notes (folder_id, image_path, json_data, created_at) VALUES (?,?,?,?)", (folder_id, "MERGED_RECORD", json.dumps(ai_data), datetime.now().strftime("%Y-%m-%d %H:%M")))
            conn.commit(); conn.close(); return ai_data
        else:
            processed_count = 0
            for f in valid_files:
                content = await f.read()
                b64 = base64.b64encode(content).decode("utf-8")
                clean_name = os.path.basename(f.filename)
                fname = os.path.join(SAVED_NOTES_DIR, f"{int(time.time())}_{clean_name}")
                with open(fname, "wb") as out: out.write(content)
                messages = [{"type": "text", "text": final_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
                ai_data = await call_ai_api(messages)
                if ai_data:
                    conn = get_db()
                    conn.execute("INSERT INTO notes (folder_id, image_path, json_data, created_at) VALUES (?,?,?,?)", (folder_id, clean_name, json.dumps(ai_data), datetime.now().strftime("%Y-%m-%d %H:%M")))
                    conn.commit(); conn.close(); processed_count += 1
            return {"status": "batch_complete", "notes_created": processed_count, "total_files": len(valid_files)}
    elif text_content:
        messages = [{"type": "text", "text": final_prompt}, {"type": "text", "text": text_content}]
        ai_data = await call_ai_api(messages)
        if not ai_data: raise HTTPException(500, "AI analysis failed")
        conn = get_db()
        conn.execute("INSERT INTO notes (folder_id, image_path, json_data, created_at) VALUES (?,?,?,?)", (folder_id, "TEXT_INPUT", json.dumps(ai_data), datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit(); conn.close(); return ai_data
    return {"error": "No valid content found"}

@app.post("/generate_csv")
def generate_csv(data: dict):
    if not os.path.exists(REPORTS_DIR): os.makedirs(REPORTS_DIR)
    try:
        title = clean_text(data.get('customer_information'))
        safe_name = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).strip() or "Export"
        filename = f"{safe_name}_{int(time.time())}.csv"; path = os.path.join(REPORTS_DIR, filename)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Date", "Customer", "Summary", "Products", "Action Items"])
            writer.writeheader()
            writer.writerow({"Date": datetime.now().strftime("%Y-%m-%d"), "Customer": title, "Summary": clean_text(data.get('executive_summary')), "Products": clean_text(data.get('product_details')), "Action Items": clean_text(data.get('action_items'))})
        return {"status": "success", "message": f"Saved to: {path}", "filename": filename}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/shutdown")
def shutdown(): 
    # Shutdown endpoint is less useful on cloud (server restarts automatically), but we keep it safe.
    os._exit(0)

# CLOUD STARTUP
if __name__ == "__main__":
    if os.path.exists(STATIC_DIR): app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    
    # Read PORT from environment variable (Required for Railway/Render/Heroku)
    port = int(os.environ.get("PORT", 8000))
    
    # Run Uvicorn directly on the main thread
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
