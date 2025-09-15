import logging
import os
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.db import connection
from django.contrib.sessions.models import Session
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
from .models import Message, Document, Embedding
from .utils import *

logger = logging.getLogger(__name__)

def ping(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            transcript = data.get("transcript")
            if transcript:
                print(f"🎤 Transcript received: {transcript}")  # ✅ Appears in Django terminal
        except Exception as e:
            print(f"⚠️ Error parsing transcript: {e}")
    request.session["ping"] = "pong"
    return JsonResponse({
        "ok": True,
        "session_key": request.session.session_key,
        "has_ping": "ping" in request.session
    })

# ------------------ Debugging & Logging ------------------

def debug_database(request):
    """Debug view to check database status"""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

        session_count = Session.objects.count()

        debug_info = {
            'database_connected': True,
            'tables_found': [table[0] for table in tables],
            'session_table_exists': 'django_session' in [table[0] for table in tables],
            'session_count': session_count,
            'database_path': connection.settings_dict.get('NAME'),
            'debug_mode': os.environ.get('DEBUG', 'False'),
        }

        logger.info(f"Database debug info: {debug_info}")
        return JsonResponse(debug_info)

    except Exception as e:
        error_info = {
            'database_connected': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'database_path': connection.settings_dict.get('NAME'),
        }
        logger.error(f"Database error: {error_info}")
        return JsonResponse(error_info, status=500)


def enhanced_login_view(request):
    """Enhanced login view with debugging"""
    logger.info(f"Login view accessed - Method: {request.method}")
    logger.info(f"Request path: {request.path}")
    logger.info(f"Session key exists: {hasattr(request, 'session')}")

    try:
        if hasattr(request, 'session'):
            request.session['test_key'] = 'test_value'
            logger.info("Session test write successful")

        # You can integrate your existing login logic here
        # Or keep this as a separate debug-only view

        return login_view(request)  # Call the existing login view logic

    except Exception as e:
        logger.error(f"Login view error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

# ------------------ Auth Views ------------------

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

# ------------------ Chat Views ------------------

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
            # Get conversation history BEFORE saving the new user message
            # This ensures the current question isn't included in the history
            conversation_history = get_conversation_history(user_uid)

            if mode == 'rag':
                # RAG Mode - Search for relevant document chunks
                context_chunks = search_similar(user_message, user_uid)
                if context_chunks:
                    # Documents found - use RAG with context and conversation history
                    context = "\n\n".join(context_chunks)
                    prompt = format_rag_prompt(conversation_history, context, user_message)
                else:
                    # No relevant documents found - inform user but maintain conversation context
                    prompt = format_no_docs_prompt(conversation_history, user_message)
            else:
                # Chat Mode - General conversation with history context
                prompt = format_chat_prompt(conversation_history, user_message)

            # Save user message to database
            Message.objects.create(user_uid=user_uid, text=user_message, sender='user')

            # Get response from Gemini API
            bot_reply = ask_gemini(prompt)

            # Save bot response to database
            Message.objects.create(user_uid=user_uid, text=bot_reply, sender='bot')

            return JsonResponse({'reply': bot_reply})

        except Exception as e:
            logger.error(f"Error in chat view: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)

    # GET request - render chat page with message history
    messages_list = Message.objects.filter(user_uid=user_uid).order_by('id')
    return render(request, 'chat.html', {
        'chat_messages': messages_list,
        'rag_on': mode == 'rag',
        'email': request.session.get('email')
    })

# ------------------ Document Views ------------------

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

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    filepath = os.path.join(settings.MEDIA_ROOT, file.name)

    # Save uploaded file temporarily
    with open(filepath, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    # Extract text from the file
    text = extract_text(filepath)
    if not text.strip():
        messages.error(request, 'Could not extract text from the file')
        os.remove(filepath)
        return redirect('chat')

    # Create document record in database
    doc = Document.objects.create(
        user_uid=request.session['user_uid'],
        filename=file.name,
        content=text
    )

    # Create embeddings for text chunks
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

    # Clean up temporary file
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


@require_http_methods(["POST"])
def delete_document(request, doc_id):
    if 'user_uid' not in request.session:
        return JsonResponse({'error': 'Not authenticated'}, status=401)

    try:
        doc = Document.objects.get(id=doc_id, user_uid=request.session['user_uid'])
        doc.delete()
        return JsonResponse({'success': True})
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found'}, status=404)


def clear_documents(request):
    if 'user_uid' not in request.session:
        return redirect('login')

    Document.objects.filter(user_uid=request.session['user_uid']).delete()
    messages.success(request, 'All documents deleted successfully!')
    return redirect('chat')

# ------------------ History Views ------------------

def search_history(request):
    """View to display user's search/chat history"""
    if 'user_uid' not in request.session:
        return redirect('login')

    user_uid = request.session['user_uid']
    messages_qs = Message.objects.filter(user_uid=user_uid).order_by('-id')

    # Group messages into conversations
    conversations = []
    current_conversation = []

    for i, msg in enumerate(messages_qs):
        current_conversation.append(msg)
        # Create a new conversation every 20 messages or at the end
        if len(current_conversation) >= 20 or i == len(messages_qs) - 1:
            if current_conversation:
                conversations.append(list(reversed(current_conversation)))
            current_conversation = []

    return render(request, 'search_history.html', {
        'conversations': conversations,
        'email': request.session.get('email')
    })


def clear_history(request):
    if 'user_uid' not in request.session:
        return redirect('login')

    Message.objects.filter(user_uid=request.session['user_uid']).delete()
    messages.success(request, 'Chat history cleared successfully!')
    return redirect('chat')

# ------------------ Mode Views ------------------

def set_mode(request):
    if request.method == 'POST':
        mode = request.POST.get('mode', 'chat')
        if mode in ['chat', 'rag']:
            request.session['mode'] = mode
    return redirect('chat')

# ------------------ SpeechRecognition Views ------------------

logger = logging.getLogger(__name__)

@csrf_exempt
def speech_to_text(request):
    """
    Accept uploaded audio (webm/ogg/mp4) -> convert to WAV using pydub/ffmpeg ->
    transcribe with speech_recognition (Google Web Speech) and return JSON.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    if "audio" not in request.FILES:
        logger.warning("No audio file in request")
        return JsonResponse({"error": "No audio uploaded"}, status=400)

    audio_file = request.FILES["audio"]
    logger.info(f"Received audio upload: name={audio_file.name}, size={audio_file.size}")

    # Ensure pydub can find ffmpeg. If ffmpeg is not on PATH, set path manually below.
    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning("ffmpeg not found on PATH. pydub may fail to convert audio.")
        # Optionally set AudioSegment.converter to a specific location:
        # AudioSegment.converter = r"C:\path\to\ffmpeg.exe"
    else:
        AudioSegment.converter = ffmpeg_path
        logger.info(f"ffmpeg path set to: {ffmpeg_path}")

    # Save uploaded file to a temp file
    in_tmp_path = None
    out_tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as in_tmp:
            for chunk in audio_file.chunks():
                in_tmp.write(chunk)
            in_tmp_path = in_tmp.name
        logger.info(f"Saved uploaded audio to temp: {in_tmp_path}")

        # Convert to WAV using pydub (requires ffmpeg)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_tmp:
            out_tmp_path = out_tmp.name

        try:
            logger.info("Attempting to convert uploaded audio to WAV with pydub...")
            audio_seg = AudioSegment.from_file(in_tmp_path)
            audio_seg = audio_seg.set_channels(1)       # mono
            audio_seg = audio_seg.set_frame_rate(16000) # 16kHz
            audio_seg.export(out_tmp_path, format="wav")
            logger.info(f"Exported WAV to {out_tmp_path}")
        except Exception as conv_err:
            logger.exception("Conversion error with pydub/ffmpeg")
            # cleanup
            if in_tmp_path and os.path.exists(in_tmp_path):
                os.remove(in_tmp_path)
            if out_tmp_path and os.path.exists(out_tmp_path):
                os.remove(out_tmp_path)
            return JsonResponse({"error": f"Audio conversion failed: {conv_err}"}, status=500)

        # Transcribe with speech_recognition
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(out_tmp_path) as source:
                audio_data = recognizer.record(source)

            logger.info("Calling recognizer.recognize_google()")
            transcript = recognizer.recognize_google(audio_data)
            logger.info(f"Transcription success: {transcript}")

            # clean up temp files
            if os.path.exists(in_tmp_path):
                os.remove(in_tmp_path)
            if os.path.exists(out_tmp_path):
                os.remove(out_tmp_path)

            return JsonResponse({"text": transcript})
        except sr.UnknownValueError:
            logger.warning("SpeechRecognition could not understand audio")
            if os.path.exists(in_tmp_path): os.remove(in_tmp_path)
            if os.path.exists(out_tmp_path): os.remove(out_tmp_path)
            return JsonResponse({"text": "Could not understand audio"}, status=400)
        except sr.RequestError as e:
            logger.exception("SpeechRecognition request error")
            if os.path.exists(in_tmp_path): os.remove(in_tmp_path)
            if os.path.exists(out_tmp_path): os.remove(out_tmp_path)
            return JsonResponse({"error": f"Recognition service error: {e}"}, status=500)
        except Exception as e:
            logger.exception("Unexpected error during recognition")
            if os.path.exists(in_tmp_path): os.remove(in_tmp_path)
            if os.path.exists(out_tmp_path): os.remove(out_tmp_path)
            return JsonResponse({"error": f"Recognition failed: {e}"}, status=500)

    except Exception as e:
        logger.exception("Error handling uploaded file")
        if in_tmp_path and os.path.exists(in_tmp_path):
            os.remove(in_tmp_path)
        if out_tmp_path and os.path.exists(out_tmp_path):
            os.remove(out_tmp_path)
        return JsonResponse({"error": f"Server error: {e}"}, status=500)
