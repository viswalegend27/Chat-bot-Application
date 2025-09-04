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
    """Get recent conversation history for context
    
    Args:
        user_uid: The user's unique identifier
        limit: Maximum number of message pairs to retrieve (default 10)
        
    Returns:
        str: Formatted conversation history as "User: ...\nAssistant: ..."
    """
    try:
        # Get recent messages, ordered by ID (most recent first)
        # Using limit*2 to get both user and assistant messages
        messages = Message.objects.filter(user_uid=user_uid).order_by('-id')[:limit*2]
        
        if not messages:
            return ""
        
        # Reverse to get chronological order (oldest first)
        messages = list(reversed(messages))
        
        history = []
        for msg in messages:
            role = 'User' if msg.sender == 'user' else 'Assistant'
            # Truncate very long messages to keep context manageable
            text = msg.text[:500] + "..." if len(msg.text) > 500 else msg.text
            history.append(f"{role}: {text}")
        
        return '\n'.join(history)
        
    except Exception as e:
        print(f'Error getting conversation history: {e}')
        return ""

def format_chat_prompt(conversation_history, user_message):
    """Format prompt for regular chat mode with conversation context
    
    Args:
        conversation_history: Previous conversation as formatted string
        user_message: Current user's message
        
    Returns:
        str: Formatted prompt for the LLM
    """
    if conversation_history:
        return f"""Previous conversation:
{conversation_history}

Current question: {user_message}

Please provide a helpful response considering the conversation context above. Be natural and conversational."""
    else:
        return user_message

def format_rag_prompt(conversation_history, context, user_message):
    """Format prompt for RAG mode with conversation context and document context
    
    Args:
        conversation_history: Previous conversation as formatted string  
        context: Relevant document chunks joined together
        user_message: Current user's message
        
    Returns:
        str: Formatted RAG prompt for the LLM
    """
    prompt = "You are a helpful assistant that answers questions based on provided document context."
    
    if conversation_history:
        prompt += f"""

Previous conversation:
{conversation_history}"""
    
    prompt += f"""

Context from uploaded documents:
{context}

Current question: {user_message}

Instructions:
- Answer the current question based primarily on the provided document context
- Use the conversation history to better understand the context and any references
- If the current question refers to something from our previous conversation, acknowledge that connection
- If the document context doesn't contain sufficient information to answer the current question, clearly state this
- Be conversational and natural in your response"""
    
    return prompt

def format_no_docs_prompt(conversation_history, user_message):
    """Format prompt when no documents are available for RAG mode
    
    Args:
        conversation_history: Previous conversation as formatted string
        user_message: Current user's message
        
    Returns:
        str: Formatted prompt explaining no documents are available
    """
    base_message = f"""I don't have any relevant documents uploaded to answer this question: "{user_message}"

To use Document Q&A mode effectively, please upload some documents first (PDF, DOCX, or TXT files)."""
    
    if conversation_history:
        return f"""Previous conversation:
{conversation_history}

Current situation: {base_message}

However, I can see our conversation history above. If you'd like to:
- Continue discussing something from our previous conversation
- Ask a general question that doesn't require document context
- Switch to regular chat mode

I'd be happy to help with that! Otherwise, please upload some documents to enable document-based Q&A."""
    
    return base_message

def ask_gemini(text):
    """Send request to Gemini API and get response
    
    Args:
        text: The prompt text to send to Gemini
        
    Returns:
        str: Cleaned response from Gemini or error message
    """
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={settings.GOOGLE_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json={'contents': [{'parts': [{'text': text}]}]},
            timeout=15
        )
        
        if response.status_code != 200:
            print(f'Gemini API error: {response.status_code} - {response.text}')
            return "Sorry, I couldn't process your request right now. Please try again."
        
        data = response.json()
        
        if 'candidates' not in data or not data['candidates']:
            print(f'No candidates in Gemini response: {data}')
            return "Sorry, I couldn't generate a response. Please try again."
        
        raw_response = data['candidates'][0]['content']['parts'][0]['text']
        return clean_ai_response(raw_response)
        
    except requests.exceptions.Timeout:
        print('Gemini API timeout')
        return "Sorry, the request timed out. Please try again."
    except Exception as e:
        print(f'Gemini error: {e}')
        return "Sorry, I couldn't process your request right now. Please try again."

def get_embedding(text):
    """Get text embedding from Google's embedding API
    
    Args:
        text: Text to get embedding for
        
    Returns:
        numpy.array: Embedding vector or None if error
    """
    try:
        if not text.strip():
            return None
            
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={settings.GOOGLE_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json={'model': 'models/text-embedding-004', 'content': {'parts': [{'text': text}]}},
            timeout=15
        )
        
        if response.status_code != 200:
            print(f'Embedding API error: {response.status_code} - {response.text}')
            return None
        
        data = response.json()
        embedding_values = data.get('embedding', {}).get('values', [])
        
        if not embedding_values:
            print(f'No embedding values in response: {data}')
            return None
            
        return np.array(embedding_values, dtype=float)
        
    except requests.exceptions.Timeout:
        print('Embedding API timeout')
        return None
    except Exception as e:
        print(f'Embedding error: {e}')
        return None

def search_similar(query, user_uid, threshold=0.1):
    """Search for similar document chunks using cosine similarity
    
    Args:
        query: Search query text
        user_uid: User's unique identifier
        threshold: Minimum similarity threshold (default 0.1)
        
    Returns:
        list: List of most similar text chunks
    """
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query)
        if query_embedding is None or not query_embedding.size:
            print('Could not get query embedding')
            return []
        
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            print('Query embedding has zero norm')
            return []
        
        scored_chunks = []
        
        # Get all user documents and their embeddings
        user_documents = Document.objects.filter(user_uid=user_uid)
        if not user_documents.exists():
            print(f'No documents found for user {user_uid}')
            return []
        
        embeddings = Embedding.objects.filter(document__in=user_documents)
        if not embeddings.exists():
            print(f'No embeddings found for user {user_uid}')
            return []
        
        # Calculate similarity scores
        for embedding in embeddings:
            try:
                vector = embedding.get_vector()
                if vector is None:
                    continue
                    
                vector = np.asarray(vector, dtype=float)
                if vector.size != query_embedding.size:
                    print(f'Vector size mismatch: {vector.size} vs {query_embedding.size}')
                    continue
                
                vector_norm = np.linalg.norm(vector)
                if vector_norm == 0:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, vector) / (query_norm * vector_norm)
                
                # Only include chunks above threshold
                if similarity >= threshold:
                    scored_chunks.append((similarity, embedding.chunk))
                    
            except Exception as e:
                print(f'Error processing embedding: {e}')
                continue
        
        # Sort by similarity score (highest first) and return top K
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        result_chunks = [chunk for _, chunk in scored_chunks[:TOP_K]]
        
        print(f'Found {len(result_chunks)} relevant chunks for query: {query[:50]}...')
        return result_chunks
        
    except Exception as e:
        print(f'Error in search_similar: {e}')
        return []