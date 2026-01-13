# KEEP - Intelligent Personal Health Vault

**KEEP** is a sophisticated medical records management platform designed to help individuals and families organize, understand, and securely share their health history. It combines a secure digital vault with an **empathic AI medical assistant ("Kelly")** powered by RAG (Retrieval-Augmented Generation) technology.

> **⚠️ Important Medical Disclaimer**: KEEP uses advanced AI to summarize records and answer questions. While highly capable, it is for **informational and organizational purposes only**. This is NOT medical advice. Always consult a qualified healthcare professional.

---

## 🚀 Key Features

### 🔐 **Enterprise-Grade Authentication**
- **Supabase Auth Integration**: Secure sign-up/login via Google OAuth or Email/Password.
- **Session Management**: Robust token handling with JWT verification.
- **Security**: Automatic session timeouts and secure logout.

### 📄 **Smart Document Processing**
- **Universal Upload**: Supports PDF, Images (JPG/PNG), and text files.
- **Multi-Engine Extraction**:
  - **OCR**: Tesseract for scanned documents and images.
  - **PDF Parsing**: PyPDF2 for native digital documents.
- **Automated Analysis**: Every upload is automatically processed, indexed, and summarized.

### 🤖 **"Kelly" - Your AI Health Assistant**
- **Empathetic Persona**: Interacts with a warm, caring, and professional tone ("Kelly").
- **Context-Aware**: Knows who you are, your family members, and your medical history.
- **RAG-Powered**: Answers questions based *strictly* on your uploaded documents and authoritative medical sources.
- **Source Attribution**: Every claim is backed by citations from your records or trusted medical bodies (WHO, CDC, Mayo Clinic).

### 👨‍👩‍👧‍👦 **Family Profile Management**
- **Multi-Profile Support**: Manage records for yourself, spouse, children, or elderly parents under one account.
- **Context Switching**: Instantly switch views to see only records relevant to a specific family member.
- **Privacy**: Data separation ensures "Kelly" knows exactly whose record is being discussed.

### 🔗 **Secure Sharing with Control**
- **Granular Sharing**: Select specific documents to share (e.g., "Just my recent blood work").
- **Time-Limited Links**: Set expiry (24h, 7 days, 30 days) for temporary access.
- **Revocable Access**: Instantly revoke any share link if sent by mistake.
- **Read-Only View**: Recipients get a clean, secure view without needing an account.

### 📊 **Timeline & Analytics**
- **Visual Timeline**: A chronological feed of all health events (uploads, AI insights, shares).
- **Activity Tracking**: Keep track of when records were added or shared.

---

## 🛠️ Tech Stack

### **Frontend**
- **HTML5 / Vanilla JS**: Lightweight, fast, single-file architecture.
- **Tailwind CSS**: Modern, utility-first styling for a clean, responsive UI.
- **Supabase JS Client**: For Authentication and real-time interactions.
- **FontAwesome**: Rich iconography.

### **Backend** (FastAPI)
- **FastAPI**: High-performance async Python web framework.
- **Authentication**: JWT validation via `python-jose` and Supabase.
- **Database**: 
  - **PostgreSQL** (via Supabase) for user/profile metadata.
  - **ChromaDB** for vector storage (semantic search).
- **Task Queue**: Background document processing.

### **AI & RAG Pipeline**
- **Primary LLM**: **OpenRouter** (accessing GPT-4o / Claude 3.5 Sonnet / Llama 3) for high-quality reasoning.
- **Fallback LLM**: **Google Gemini 2.0 Flash** for speed and redundancy.
- **Embeddings**: **Google Gemini Text Embedding 004** for state-of-the-art vector search.
- **Knowledge Base**: Curated local vector store of trusted medical information (CDC, WHO, NIH).

---

## 🏗️ Architecture

```mermaid
graph TD
    User[User] -->|Browser| Frontend[Frontend (HTML/JS)]
    Frontend -->|Auth (JWT)| Supabase[Supabase Auth]
    Frontend -->|API Requests| Backend[FastAPI Backend]
    
    subgraph Backend Services
        Backend -->|Store Metadata| PG[(PostgreSQL)]
        Backend -->|Vector Search| Chroma[(ChromaDB)]
        Backend -->|Generate| AI_LLM[OpenRouter/Gemini]
        Backend -->|Embed| AI_Embed[Gemini Embeddings]
    end
    
    subgraph Processing Pipeline
        Upload[File Upload] --> OCR[Tesseract/PyPDF2]
        OCR --> Chunking[Text Chunking]
        Chunking --> AI_Embed
        AI_Embed --> Chroma
    end
```

---

## 🔧 Setup & Installation

### Prerequisites
- **Python 3.10+**
- **Supabase Project** (for Auth & DB)
- **Google API Key** (for Embeddings/Gemini)
- **OpenRouter API Key** (Optional, for premium models)
- **Tesseract OCR** installed on system.

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/keep.git
cd keep/backend
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in `/backend` with the following:

```ini
# Core
SECRET_KEY=your_random_secret_string
ENVIRONMENT=development

# AI Providers
GOOGLE_API_KEY=your_google_ai_key
OPENROUTER_API_KEY=your_openrouter_key (optional)

# Supabase (Required for Auth)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# Database (Local dev uses SQLite by default)
DATABASE_URL=sqlite:///./keep.db
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads
```

### 3. Initialize Medical Knowledge Base (Optional)
To give "Kelly" medical knowledge context:
```bash
python -m backend.medical_kb_ingestion
```

### 4. Run the Application
```bash
# Start Backend
uvicorn backend.main:app --reload --port 8000

# Start Frontend
# Just open frontend/index.html in your browser!
# Or verify via http://localhost:8000/docs
```

---

## 📚 API Documentation

Once running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Core Endpoints
| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/auth/sync` | Sync Supabase session with backend |
| `GET` | `/profiles` | List family profiles |
| `POST` | `/records/upload` | Upload & Process Document |
| `GET` | `/timeline` | Get user activity feed |
| `POST` | `/chat` | Chat with "Kelly" (RAG) |
| `POST` | `/shares` | Generate secure share link |

---

## 🛡️ Security Features

- **Data Isolation**: Users can strictly only access their own data.
- **Encryption**: All connections over TLS.
- **Ephemeral Sharing**: Share links automatically expire.
- **Fail-Safe Auth**: Frontend gracefully handles network failures without locking users out.

---

## 🔮 Roadmap

- [ ] **Cloud Storage**: Move from local `uploads/` to AWS S3 / Supabase Storage.
- [ ] **Mobile App**: React Native wrapper for the frontend.
- [ ] **Integrations**: Connect with Apple Health / Google Fit.
- [ ] **Audit Logs**: Detailed CSV export of who accessed what and when.

---

**Built with ❤️ for a healthier future.**
# KEEP
