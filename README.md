# 🤖 AI Chat & Document Q&A System

A powerful Django-based web application that combines conversational AI with intelligent document search. Chat with AI about anything, or upload your documents and get precise answers from your own content!

## ✨ What Makes This Special?

### 🧠 Dual-Mode Intelligence
- **General Chat Mode**: Have natural conversations with Google's Gemini AI
- **Document Q&A Mode**: Upload your files and get answers directly from your content
- **Smart Context Switching**: Seamlessly switch between modes while maintaining conversation flow

### 📄 Document Processing Power
- **Multi-Format Support**: PDF, DOCX, and TXT files
- **Intelligent Chunking**: Documents are split into searchable segments
- **Vector Embeddings**: Uses Google's latest embedding model for semantic search
- **Contextual Retrieval**: Finds the most relevant information for your questions

### 💬 Conversation Features
- **Memory That Works**: Maintains conversation context across all interactions
- **Follow-up Questions**: Ask "Can you elaborate?" or "What about X?" and it remembers
- **Smart Prompting**: Different conversation styles for chat vs document queries
- **History Management**: View, search, and clear your conversation history

## 🚀 Quick Start Guide

### Prerequisites

Make sure you have these installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git

### 1. Clone & Setup

```bash
# Get the code
git clone <your-repository-url>
cd ai-chat-document-qa

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in your project root:

```env
# Google AI API Key (required)
GOOGLE_API_KEY=your_google_ai_api_key_here

# Firebase Authentication (required)
FIREBASE_API_KEY=your_firebase_api_key_here

# Django Settings
DEBUG=True
SECRET_KEY=your_django_secret_key_here
```

**🔑 Getting API Keys:**

- **Google AI API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) and create a free API key
- **Firebase API Key**: Create a project at [Firebase Console](https://console.firebase.google.com/), enable Authentication, and copy your Web API key

### 3. Database Setup

```bash
# Create database tables
python manage.py makemigrations
python manage.py migrate

# Create a superuser (optional)
python manage.py createsuperuser
```

### 4. Launch the Application

```bash
python manage.py runserver
```

Visit `http://localhost:8000` and you're ready to go! 🎉

## 🎯 How to Use

### Getting Started
1. **Sign Up**: Create an account with email and password
2. **Choose Your Mode**: Switch between Chat and Document Q&A modes
3. **Start Chatting**: Ask anything or upload documents to get started

### General Chat Mode
- Ask any question: "What is quantum computing?"
- Have conversations: "Can you explain that differently?"
- Get creative: "Write me a poem about coding"

### Document Q&A Mode
1. **Upload Documents**: Click upload and select PDF, DOCX, or TXT files
2. **Wait for Processing**: Documents are automatically chunked and indexed
3. **Ask Questions**: "What does the contract say about payment terms?"
4. **Follow Up**: "Can you find more details about that?"

### Pro Tips 💡
- **Use Follow-ups**: The system remembers your conversation, so ask follow-up questions naturally
- **Mix Modes**: Switch between modes mid-conversation - context is preserved
- **Manage Documents**: View uploaded documents and delete ones you no longer need
- **Clear History**: Start fresh anytime by clearing your conversation history

## 🏗️ Project Architecture

### Core Components
```
📁 Project Structure
├── 🌐 views.py          # Web interface and API endpoints
├── 🛠️ utils.py           # AI processing and document handling
├── 📊 models.py         # Database structure
├── 🎨 templates/        # HTML templates
├── 📁 static/           # CSS, JS, images
└── ⚙️ settings.py       # Configuration
```

### Technology Stack
- **Backend**: Django (Python web framework)
- **Database**: SQLite (easily upgradeable to PostgreSQL)
- **AI Models**: Google Gemini 2.0 Flash + Text Embedding 004
- **Authentication**: Firebase Auth
- **Document Processing**: PyMuPDF (PDF), python-docx (Word)
- **Vector Search**: NumPy-based cosine similarity

### Data Flow
1. **User Input** → Django Views
2. **Context Retrieval** → Database + Vector Search
3. **Prompt Generation** → Smart prompt formatting
4. **AI Processing** → Google Gemini API
5. **Response** → User Interface

## 🔧 Advanced Configuration

### Customizing Chunk Size
Edit `utils.py`:
```python
CHUNK_SIZE = 800  # Increase for longer chunks
TOP_K = 3         # Number of relevant chunks to retrieve
```

### Database Upgrade
For production, switch to PostgreSQL:
```python
# In settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db_name',
        # ... other settings
    }
}
```

### Scaling Considerations
- Add Redis for caching embeddings
- Use Celery for background document processing
- Implement rate limiting for API calls
- Add file size limits for uploads

## 🐛 Troubleshooting

### Common Issues

**"No module named 'fitz'"**
```bash
pip install PyMuPDF
```

**"Authentication failed"**
- Check your Firebase API key in `.env`
- Ensure Firebase Authentication is enabled in your project

**"Gemini API error"**
- Verify your Google AI API key
- Check if you've exceeded rate limits
- Ensure the API key has proper permissions

**Documents not uploading**
- Check file format (PDF, DOCX, TXT only)
- Ensure the file contains extractable text
- Check file size (large files may timeout)

### Debug Mode
Enable detailed logging by adding this to `settings.py`:
```python
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}
```

## 🤝 Contributing

We'd love your help making this better! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin amazing-feature`
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python manage.py test

# Format code
black .
```

## 🙏 Acknowledgments

- **Google AI** for the powerful Gemini and Embedding APIs
- **Firebase** for seamless authentication
- **Django Community** for the robust web framework
- **Open Source Contributors** for the amazing libraries that make this possible

## 🌐 Deploying to Render

Ready to share your AI assistant with the world? Here's how to deploy on Render (free tier available!):

### 1. Prepare for Production

Create a `requirements.txt` file:
```txt
Django==4.2.7
requests==2.31.0
numpy==1.24.3
PyMuPDF==1.23.5
python-docx==0.8.11
gunicorn==21.2.0
whitenoise==6.6.0
```

Create a `build.sh` file in your project root:
```bash
#!/usr/bin/env bash
# build.sh
set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate
```

Make it executable:
```bash
chmod +x build.sh
```

### 2. Update Django Settings

Add to your `settings.py`:
```python
import os
from pathlib import Path

# Production settings
if os.environ.get('RENDER'):
    DEBUG = False
    ALLOWED_HOSTS = ['*']  # Configure this properly for production
    
    # Database for production (SQLite works fine for small apps)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Middleware (add WhiteNoise)
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Add this line
    # ... your other middleware
]
```

### 3. Deploy to Render

1. **Push to GitHub**: Make sure your code is on GitHub
2. **Go to Render**: Visit [render.com](https://render.com) and create an account
3. **New Web Service**: Click "New" → "Web Service"
4. **Connect Repository**: Select your GitHub repository
5. **Configure Service**:
   - **Name**: `your-ai-chat-app`
   - **Environment**: `Python 3`
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn your_project_name.wsgi:application --host 0.0.0.0 --port $PORT`

### 4. Environment Variables

In Render's dashboard, add these environment variables:
- `GOOGLE_API_KEY`: Your Google AI API key
- `FIREBASE_API_KEY`: Your Firebase API key
- `SECRET_KEY`: A secure Django secret key
- `RENDER`: `True` (to enable production settings)

### 5. Go Live! 🚀

- Click "Create Web Service"
- Wait for deployment (first deploy takes 5-10 minutes)
- Your app will be available at `https://your-app-name.onrender.com`

### 💰 Render Pricing
- **Free Tier**: Perfect for demos and small projects
- **Paid Tier**: $7/month for better performance and no sleep mode
- **Database**: Free PostgreSQL available if you need it later

### 🔧 Production Tips
- **Custom Domain**: Add your own domain in Render dashboard
- **SSL**: Automatically provided by Render
- **Monitoring**: Use Render's built-in logging and metrics
- **Scaling**: Easy horizontal scaling with paid plans

## 📞 Support

Having trouble? We're here to help!

- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: We love new ideas!
- 📖 **Documentation**: Check our wiki for detailed guides
- 💬 **Community**: Join our discussions
- 🌐 **Deployment Issues**: Check Render's excellent documentation

---

**Built with ❤️ for the AI community. Happy chatting! 🚀**
