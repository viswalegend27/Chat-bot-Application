import os, requests, numpy as np, fitz, docx, re
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Config
load_dotenv()
BASE = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-123")
app.config.update({
    "SQLALCHEMY_DATABASE_URI": "sqlite:///" + os.path.join(BASE, "chatbot.db"),
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
    "UPLOAD_FOLDER": UPLOAD_FOLDER
})
db = SQLAlchemy(app)

FIREBASE_KEY = os.getenv("FIREBASE_API_KEY")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
ALLOWED_EXT = {"pdf", "docx", "txt"}
CHUNK_SIZE = 800
TOP_K = 3

# Models
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_uid = db.Column(db.String(200))
    text = db.Column(db.Text)
    sender = db.Column(db.String(50))

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_uid = db.Column(db.String(200))
    filename = db.Column(db.String(200))
    content = db.Column(db.Text)

class Embedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doc_id = db.Column(db.Integer, db.ForeignKey("document.id"))
    chunk = db.Column(db.Text)
    vector = db.Column(db.PickleType)

# Helper Functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def chunk_text(text):
    text = text.strip()
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

def extract_text(filepath):
    ext = filepath.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            return "\n".join(page.get_text("text") for page in fitz.open(filepath))
        elif ext == "docx":
            return "\n".join(p.text for p in docx.Document(filepath).paragraphs if p.text.strip())
        elif ext == "txt":
            with open(filepath, encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Extract error: {e}")
    return ""

def clean_ai_response(response_text):
    if not response_text:
        return response_text
    
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text.strip())
    text = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', text)
    text = re.sub(r'\*\*\*([^*:]+):\*\*', r'**\1:**', text)
    text = re.sub(r'\*{3,}', '**', text)
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    text = re.sub(r'\*\*([^*:]+):\*\*([^\n])', r'**\1:** \2', text)
    text = re.sub(r':\s*([A-Z])', r': \1', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def firebase_auth(endpoint, email, password):
    try:
        response = requests.post(
            f"{endpoint}?key={FIREBASE_KEY}",
            json={"email": email, "password": password, "returnSecureToken": True},
            timeout=10
        )
        return response.json()
    except Exception as e:
        print(f"Firebase auth error: {e}")
        return {"error": {"message": "Authentication failed"}}

def ask_gemini(text):
    try:
        formatted_prompt = f"""Please provide a clean, well-formatted response to the following question. 
Use proper markdown formatting with clear headings and bullet points where appropriate.

Question: {text}"""
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": formatted_prompt}]}]},
            timeout=15
        )
        data = response.json()
        raw_response = data["candidates"][0]["content"]["parts"][0]["text"]
        return clean_ai_response(raw_response)
        
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Sorry, I couldn't process your request right now."

def get_embedding(text):
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            json={"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}},
            timeout=15
        )
        data = response.json()
        return np.array(data.get("embedding", {}).get("values", []), dtype=float)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def search_similar(query, user_uid):
    query_embedding = get_embedding(query)
    if query_embedding is None or not query_embedding.size:
        return []
    
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    
    scored_chunks = []
    embeddings = Embedding.query.join(Document).filter(Document.user_uid == user_uid).all()
    
    for embedding in embeddings:
        vector = np.asarray(embedding.vector, dtype=float)
        if vector.size != query_embedding.size:
            continue
        
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0:
            continue
            
        similarity = np.dot(query_embedding, vector) / (query_norm * vector_norm)
        scored_chunks.append((similarity, embedding.chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:TOP_K]]

# Routes
@app.route("/")
def home():
    return redirect(url_for("chat" if "user_uid" in session else "login"))

@app.route("/documents")
def documents():
    if "user_uid" not in session:
        return redirect(url_for("login"))
    
    user_docs = Document.query.filter_by(user_uid=session["user_uid"]).all()
    docs_with_chunks = []
    
    for doc in user_docs:
        chunk_count = Embedding.query.filter_by(doc_id=doc.id).count()
        docs_with_chunks.append({
            'id': doc.id,
            'filename': doc.filename,
            'chunk_count': chunk_count,
            'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        })
    
    return jsonify({"documents": docs_with_chunks})

@app.route("/delete_document/<int:doc_id>", methods=["POST"])
def delete_document(doc_id):
    if "user_uid" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    doc = Document.query.filter_by(id=doc_id, user_uid=session["user_uid"]).first()
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    # Delete associated embeddings
    Embedding.query.filter_by(doc_id=doc_id).delete()
    # Delete document
    db.session.delete(doc)
    db.session.commit()
    
    return jsonify({"success": True})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "user_uid" not in session:
        return redirect(url_for("login"))
    
    Message.query.filter_by(user_uid=session["user_uid"]).delete()
    db.session.commit()
    flash("Chat history cleared successfully!")
    return redirect(url_for("chat"))


@app.route("/clear_documents", methods=["POST"])
def clear_documents():
    if "user_uid" not in session:
        return redirect(url_for("login"))
    
    user_uid = session["user_uid"]
    docs = Document.query.filter_by(user_uid=user_uid).all()
    doc_ids = [doc.id for doc in docs]
    
    # Delete all embeddings for user's documents
    Embedding.query.filter(Embedding.doc_id.in_(doc_ids)).delete(synchronize_session=False)
    # Delete all documents for user
    Document.query.filter_by(user_uid=user_uid).delete()
    db.session.commit()
    flash("All document history deleted successfully!")
    return redirect(url_for("chat"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        result = firebase_auth(
            "https://identitytoolkit.googleapis.com/v1/accounts:signUp",
            request.form["email"], 
            request.form["password"]
        )
        if "error" in result:
            flash(result["error"]["message"])
            return redirect(url_for("signup"))
        flash("Account created successfully! Please log in.")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        result = firebase_auth(
            "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
            request.form["email"], 
            request.form["password"]
        )
        if "error" in result:
            flash(result["error"]["message"])
            return redirect(url_for("login"))
        
        session["user_uid"] = result["localId"]
        session["email"] = request.form["email"]
        return redirect(url_for("chat"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/upload", methods=["POST"])
def upload():
    if "user_uid" not in session:
        return redirect(url_for("login"))
    
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        flash("Please select a valid file (PDF, DOCX, or TXT)")
        return redirect(url_for("chat"))
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    
    text = extract_text(filepath)
    if not text.strip():
        flash("Could not extract text from the file")
        os.remove(filepath)
        return redirect(url_for("chat"))
    
    user_uid = session["user_uid"]
    doc = Document(user_uid=user_uid, filename=filename, content=text)
    db.session.add(doc)
    db.session.flush()
    
    chunks = chunk_text(text)
    success_count = 0
    for chunk in chunks:
        if chunk.strip():
            vector = get_embedding(chunk)
            if vector is not None and vector.size > 0:
                embedding = Embedding(doc_id=doc.id, chunk=chunk, vector=vector)
                db.session.add(embedding)
                success_count += 1
    
    db.session.commit()
    os.remove(filepath)
    
    if success_count > 0:
        flash(f"Document uploaded successfully! Created {success_count} searchable chunks.")
    else:
        flash("Document uploaded but no embeddings could be created.")
    
    return redirect(url_for("chat"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "user_uid" not in session:
        return redirect(url_for("login"))
    
    user_uid = session["user_uid"]
    mode = session.get("mode", "chat")
    
    if request.method == "POST":
        user_message = request.form.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        user_msg_obj = Message(user_uid=user_uid, text=user_message, sender="user")
        db.session.add(user_msg_obj)
        
        if mode == "rag":
            context_chunks = search_similar(user_message, user_uid)
            if context_chunks:
                context = "\n\n".join(context_chunks)
                prompt = f"Based on the following context, answer the question. If the context doesn't contain relevant information, say so.\n\nContext:\n{context}\n\nQuestion: {user_message}"
            else:
                prompt = f"I don't have any relevant documents uploaded to answer this question: {user_message}\n\nPlease upload some documents first to use Document Q&A mode."
        else:
            prompt = user_message
        
        bot_reply = ask_gemini(prompt)
        
        bot_msg_obj = Message(user_uid=user_uid, text=bot_reply, sender="bot")
        db.session.add(bot_msg_obj)
        db.session.commit()
        
        return jsonify({"reply": bot_reply})
    
    messages = Message.query.filter_by(user_uid=user_uid).all()
    return render_template("chat.html", 
                        messages=messages, 
                        rag_on=(mode == "rag"), 
                        email=session.get("email"))

@app.route("/set_mode", methods=["POST"])
def set_mode():
    mode = request.form.get("mode", "chat")
    if mode in ["chat", "rag"]:
        session["mode"] = mode
    return redirect(url_for("chat"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)