import os, requests, numpy as np, fitz, docx, re
from django.conf import settings
from .models import Message, Document, Embedding

ALLOWED_EXT = {'pdf', 'docx', 'txt'}
CHUNK_SIZE = 800
TOP_K = 3

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def chunk_text(text):
    text = text.strip()
    return [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'pdf':
            return '\n'.join(page.get_text('text') for page in fitz.open(filepath))
        elif ext == 'docx':
            return '\n'.join(p.text for p in docx.Document(filepath).paragraphs if p.text.strip())
        elif ext == 'txt':
            with open(filepath, encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f'Extract error: {e}')
        return ''

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
            f"{endpoint}?key={settings.FIREBASE_API_KEY}",
            json={'email': email, 'password': password, 'returnSecureToken': True},
            timeout=10
        )
        return response.json()
    except Exception as e:
        print(f'Firebase auth error: {e}')
        return {'error': {'message': 'Authentication failed'}}

def get_conversation_history(user_uid, limit=10):
    """Get recent conversation history for context"""
    messages = Message.objects.filter(user_uid=user_uid).order_by('-timestamp')[:limit*2]
    messages = list(reversed(messages))  # Reverse to chronological order
    
    history = []
    for msg in messages:
        role = 'User' if msg.sender == 'user' else 'Assistant'
        history.append(f"{role}: {msg.text}")
    
    return '\n'.join(history)

def format_chat_prompt(conversation_history, user_message):
    """Format prompt for regular chat mode with conversation context"""
    if conversation_history:
        return f"""Previous conversation:
{conversation_history}

Current question: {user_message}

Please provide a helpful response considering the conversation context."""
    else:
        return user_message

def format_rag_prompt(conversation_history, context, user_message):
    """Format prompt for RAG mode with conversation context"""
    prompt = f"""Based on the following context from uploaded documents, answer the question."""
    
    if conversation_history:
        prompt += f"""

Previous conversation:
{conversation_history}"""
    
    prompt += f"""

Context from documents:
{context}

Current question: {user_message}

Please answer based on the provided context. If the context doesn't contain relevant information for the current question, say so clearly."""
    
    return prompt

def format_no_docs_prompt(conversation_history, user_message):
    """Format prompt when no documents are available for RAG"""
    prompt = f"I don't have any relevant documents uploaded to answer this question: {user_message}\n\nPlease upload some documents first to use Document Q&A mode."
    
    if conversation_history:
        prompt = f"""Previous conversation:
{conversation_history}

{prompt}"""
    
    return prompt

def ask_gemini(text):
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={settings.GOOGLE_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json={'contents': [{'parts': [{'text': text}]}]},
            timeout=15
        )
        data = response.json()
        raw_response = data['candidates'][0]['content']['parts'][0]['text']
        return clean_ai_response(raw_response)
    except Exception as e:
        print(f'Gemini error: {e}')
        return "Sorry, I couldn't process your request right now."

def get_embedding(text):
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={settings.GOOGLE_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json={'model': 'models/text-embedding-004', 'content': {'parts': [{'text': text}]}},
            timeout=15
        )
        data = response.json()
        return np.array(data.get('embedding', {}).get('values', []), dtype=float)
    except Exception as e:
        print(f'Embedding error: {e}')
        return None

def search_similar(query, user_uid):
    query_embedding = get_embedding(query)
    if query_embedding is None or not query_embedding.size:
        return []
    
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    
    scored_chunks = []
    user_documents = Document.objects.filter(user_uid=user_uid)
    embeddings = Embedding.objects.filter(document__in=user_documents)
    
    for embedding in embeddings:
        vector = embedding.get_vector()
        if vector is None:
            continue
            
        vector = np.asarray(vector, dtype=float)
        if vector.size != query_embedding.size:
            continue
        
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0:
            continue
        
        similarity = np.dot(query_embedding, vector) / (query_norm * vector_norm)
        scored_chunks.append((similarity, embedding.chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:TOP_K]]