import os
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from .models import Message, Document, Embedding
from .utils import *
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_http_methods

def get_conversation_history(user_uid, limit=10):
    """Get recent conversation history for context"""
    messages = Message.objects.filter(user_uid=user_uid).order_by('-id')[:limit*2]
    history = []
    for msg in reversed(messages):
        if msg.sender == 'user':
            history.append(f"User: {msg.text}")
        else:
            history.append(f"Assistant: {msg.text}")
    return "\n".join(history)

def format_rag_prompt(conversation_history, context, user_message):
    """Format RAG prompt with conversation history"""
    return f"""Previous conversation:
{conversation_history}

Based on the following context, answer the current question:

Context:
{context}

Current question: {user_message}

Please provide a helpful response based on the context and conversation history."""

def format_no_docs_prompt(conversation_history, user_message):
    """Format prompt when no documents are available"""
    return f"""Previous conversation:
{conversation_history}

Current question: {user_message}

I don't have any relevant documents uploaded to answer this question. Please upload some documents first to use Document Q&A mode."""

def format_chat_prompt(conversation_history, user_message):
    """Format general chat prompt with conversation history"""
    return f"""Previous conversation:
{conversation_history}

Current question: {user_message}

Please provide a helpful response considering our conversation history."""

def home(request):
    if 'user_uid' in request.session:
        return redirect('chat')
    return redirect('login')

def signup(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        result = firebase_auth(
            "https://identitytoolkit.googleapis.com/v1/accounts:signUp",
            email, password
        )
        
        if 'error' in result:
            messages.error(request, result['error']['message'])
            return render(request, 'signup.html')
        
        messages.success(request, 'Account created successfully! Please log in.')
        return redirect('login')
    
    return render(request, 'signup.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        result = firebase_auth(
            "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
            email, password
        )
        
        if 'error' in result:
            messages.error(request, result['error']['message'])
            return render(request, 'login.html')
        
        request.session['user_uid'] = result['localId']
        request.session['email'] = email
        return redirect('chat')
    
    return render(request, 'login.html')

def logout_view(request):
    request.session.flush()
    return redirect('login')

@csrf_protect
def chat(request):
    if 'user_uid' not in request.session:
        return redirect('login')
    
    user_uid = request.session['user_uid']
    mode = request.session.get('mode', 'chat')
    
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if not user_message:
            return JsonResponse({'error': 'Empty message'}, status=400)
        
        try:
            # Get conversation history BEFORE saving current message
            conversation_history = get_conversation_history(user_uid)
            
            # Format the prompt with context (but don't include current message in history)
            if mode == 'rag':
                context_chunks = search_similar(user_message, user_uid)
                if context_chunks:
                    context = "\n\n".join(context_chunks)
                    if conversation_history:
                        prompt = f"""Previous conversation:
{conversation_history}

Based on the following context, answer the current question:

Context:
{context}

Current question: {user_message}

Please provide a helpful response based on the context and conversation history."""
                    else:
                        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {user_message}

Please provide a helpful response based on the context."""
                else:
                    if conversation_history:
                        prompt = f"""Previous conversation:
{conversation_history}

Current question: {user_message}

I don't have any relevant documents uploaded to answer this question. Please upload some documents first to use Document Q&A mode."""
                    else:
                        prompt = f"I don't have any relevant documents uploaded to answer this question: {user_message}\n\nPlease upload some documents first to use Document Q&A mode."
            else:
                if conversation_history:
                    prompt = f"""Previous conversation:
{conversation_history}

Current question: {user_message}

Please provide a helpful response considering our conversation history."""
                else:
                    prompt = user_message
            
            # Now save the user message
            user_msg = Message.objects.create(
                user_uid=user_uid,
                text=user_message,
                sender='user'
            )
            
            bot_reply = ask_gemini(prompt)
            
            # Save bot response
            bot_msg = Message.objects.create(
                user_uid=user_uid,
                text=bot_reply,
                sender='bot'
            )
            
            return JsonResponse({'reply': bot_reply})
            
        except Exception as e:
            print(f"Error in chat view: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    # Handle GET request
    messages_list = Message.objects.filter(user_uid=user_uid).order_by('id')
    return render(request, 'chat.html', {
        'chat_messages': messages_list,
        'rag_on': mode == 'rag',
        'email': request.session.get('email')
    })
    
@csrf_exempt
def upload_document(request):
    if 'user_uid' not in request.session:
        return redirect('login')
    
    if request.method != 'POST' or 'file' not in request.FILES:
        messages.error(request, 'Please select a valid file')
        return redirect('chat')
    
    file = request.FILES['file']
    if not allowed_file(file.name):
        messages.error(request, 'Please select a valid file (PDF, DOCX, or TXT)')
        return redirect('chat')
    
    # Save file temporarily
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    filepath = os.path.join(settings.MEDIA_ROOT, file.name)
    
    with open(filepath, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    # Extract text
    text = extract_text(filepath)
    if not text.strip():
        messages.error(request, 'Could not extract text from the file')
        os.remove(filepath)
        return redirect('chat')
    
    # Save document
    doc = Document.objects.create(
        user_uid=request.session['user_uid'],
        filename=file.name,
        content=text
    )
    
    # Create embeddings
    chunks = chunk_text(text)
    success_count = 0
    
    for chunk in chunks:
        if chunk.strip():
            vector = get_embedding(chunk)
            if vector is not None and vector.size > 0:
                embedding = Embedding(document=doc, chunk=chunk)
                embedding.set_vector(vector)
                embedding.save()
                success_count += 1
    
    os.remove(filepath)
    
    if success_count > 0:
        messages.success(request, f'Document uploaded! Created {success_count} searchable chunks.')
    else:
        messages.warning(request, 'Document uploaded but no embeddings could be created.')
    
    return redirect('chat')

def documents(request):
    if 'user_uid' not in request.session:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    
    user_docs = Document.objects.filter(user_uid=request.session['user_uid'])
    docs_data = []
    
    for doc in user_docs:
        chunk_count = Embedding.objects.filter(document=doc).count()
        docs_data.append({
            'id': doc.id,
            'filename': doc.filename,
            'chunk_count': chunk_count,
            'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        })
    
    return JsonResponse({'documents': docs_data})

def search_history(request):
    """View to display user's search/chat history"""
    if 'user_uid' not in request.session:
        return redirect('login')
    
    user_uid = request.session['user_uid']
    
    # Get all messages for the user
    messages = Message.objects.filter(user_uid=user_uid).order_by('-id')
    
    # Group messages into conversation threads (every 20 messages = 1 conversation)
    conversations = []
    current_conversation = []
    
    for i, msg in enumerate(messages):
        current_conversation.append(msg)
        # Create new conversation group every 20 messages
        if len(current_conversation) >= 20 or i == len(messages) - 1:
            if current_conversation:
                conversations.append(list(reversed(current_conversation)))
                current_conversation = []
    
    return render(request, 'search_history.html', {
        'conversations': conversations,
        'email': request.session.get('email')
    })

@require_http_methods(["POST"])
def delete_document(request, doc_id):
    if 'user_uid' not in request.session:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    
    try:
        doc = Document.objects.get(id=doc_id, user_uid=request.session['user_uid'])
        doc.delete()  # This will also delete associated embeddings due to CASCADE
        return JsonResponse({'success': True})
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found'}, status=404)

def clear_history(request):
    if 'user_uid' not in request.session:
        return redirect('login')
    
    Message.objects.filter(user_uid=request.session['user_uid']).delete()
    messages.success(request, 'Chat history cleared successfully!')
    return redirect('chat')

def clear_documents(request):
    if 'user_uid' not in request.session:
        return redirect('login')
    
    Document.objects.filter(user_uid=request.session['user_uid']).delete()
    messages.success(request, 'All documents deleted successfully!')
    return redirect('chat')

def set_mode(request):
    if request.method == 'POST':
        mode = request.POST.get('mode', 'chat')
        if mode in ['chat', 'rag']:
            request.session['mode'] = mode
    return redirect('chat')