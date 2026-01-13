# KEEP Backend - Complete Implementation Guide

## Overview
Complete FastAPI backend with RAG-powered AI assistant for the KEEP health platform.

## Architecture

```
Frontend (HTML/JS/Tailwind)
         ↓
    FastAPI REST API
         ↓
    ┌────┴────┐
    ↓         ↓
PostgreSQL  ChromaDB + Google Gemini
(metadata)  (vector search + AI)
```

## Features Implemented

### ✅ Complete API Coverage
Every frontend feature has a corresponding backend endpoint:

1. **Authentication** - Login, user management
2. **Profiles** - Family member profiles
3. **Records** - Upload, list, view, download, delete
4. **AI Analysis** - Automatic document analysis with Gemini
5. **RAG Chat** - Context-aware medical Q&A
6. **Sharing** - Secure, time-limited share links
7. **Timeline** - Activity feed

### ✅ RAG Pipeline
- **Document Ingestion**: OCR + PDF extraction → chunking → embedding → ChromaDB
- **Medical Knowledge Base**: 10 authoritative sources pre-ingested
- **Retrieval**: Hybrid search (user docs + medical knowledge)
- **Generation**: Google Gemini 2.5 Flash with grounded responses

### ✅ Medical Knowledge Sources
1. WHO (World Health Organization)
2. CDC (Centers for Disease Control)
3. NIH (National Institutes of Health)
4. Mayo Clinic
5. Cleveland Clinic
6. NHS (UK)
7. MedlinePlus
8. WebMD
9. UpToDate
10. American Heart Association

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: For OCR, install Tesseract:
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings:
# - DATABASE_URL (PostgreSQL connection string)
# - GOOGLE_API_KEY (from https://makersuite.google.com/app/apikey)
```

### 3. Setup Database

```bash
# Create PostgreSQL database
createdb keep_db

# Or use Supabase/hosted PostgreSQL
# Set DATABASE_URL in .env
```

### 4. Ingest Medical Knowledge Base (One-time)

```bash
# Set your GOOGLE_API_KEY in medical_kb_ingestion.py or environment
python -m backend.medical_kb_ingestion
```

This will populate ChromaDB with medical knowledge for RAG.

### 5. Run the Server

```bash
uvicorn backend.main:app --reload --port 8000
```

API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Authentication
```
POST   /api/v1/auth/login          - Login/create user
GET    /api/v1/auth/me             - Get current user
```

### Profiles
```
POST   /api/v1/profiles            - Create profile
GET    /api/v1/profiles            - List profiles
GET    /api/v1/profiles/{id}       - Get profile
```

### Records
```
POST   /api/v1/records/upload      - Upload file
GET    /api/v1/records             - List records
GET    /api/v1/records/{id}        - Get record detail
GET    /api/v1/records/{id}/file   - Download file
PUT    /api/v1/records/{id}        - Update metadata
DELETE /api/v1/records/{id}        - Delete record
```

### Chat
```
POST   /api/v1/chat                - Send message
GET    /api/v1/chat/history        - Get history
DELETE /api/v1/chat/history        - Clear history
```

### Sharing
```
POST   /api/v1/shares              - Create share
GET    /api/v1/shares              - List shares
PUT    /api/v1/shares/{id}/revoke  - Revoke share
GET    /s/{token}                  - Public share access
```

### Timeline
```
GET    /api/v1/timeline            - Get events
```

## File Structure

```
backend/
├── main.py                    # FastAPI app + all routes
├── models.py                  # SQLAlchemy ORM models
├── schemas.py                 # Pydantic request/response schemas
├── rag_pipeline.py            # RAG logic (ChromaDB + Gemini)
├── utils.py                   # OCR, PDF extraction, chunking
├── config.py                  # Settings
├── db.py                      # Database connection
├── medical_kb_ingestion.py    # One-time KB setup script
├── requirements.txt           # Dependencies
└── .env.example               # Environment template
```

## Database Schema

See `models.py` for complete schema.

**Key Tables**:
- `users` - User accounts
- `profiles` - Family members
- `records` - Health records metadata
- `record_files` - File storage info
- `record_text` - Extracted text (OCR/PDF)
- `record_analysis` - AI-generated summaries
- `chat_history` - Conversation logs
- `shares` - Share links
- `timeline_events` - Activity feed

## Frontend Integration

### Example: Upload Flow

**Frontend**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('profile_id', currentProfileId);
formData.append('document_name', 'Blood Test Results');
formData.append('record_date', '2024-01-15');

const response = await fetch('http://localhost:8000/api/v1/records/upload', {
  method: 'POST',
  body: formData
});

const data = await response.json();
// { record_id: "...", status: "Processing", message: "..." }

// Poll for completion
const checkStatus = setInterval(async () => {
  const record = await fetch(`http://localhost:8000/api/v1/records/${data.record_id}`);
  const recordData = await record.json();
  
  if (recordData.status === "Analyzed") {
    clearInterval(checkStatus);
    // Show summary: recordData.analysis
  }
}, 2000);
```

### Example: Chat

**Frontend**:
```javascript
const response = await fetch('http://localhost:8000/api/v1/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    profile_id: currentProfileId,
    message: "What do my Vitamin D levels mean?"
  })
});

const data = await response.json();
// {
//   message: "Your Vitamin D level of 28 ng/mL is in the insufficient range...",
//   sources: [
//     { type: "user_document", title: "Blood Panel Oct 2023" },
//     { type: "medical_knowledge", title: "NIH - Vitamin D" }
//   ],
//   confidence: 85.0
// }
```

## RAG Pipeline Details

### Document Processing
1. **Upload** → Save file
2. **Extract** → OCR (images) or PyPDF2 (PDFs)
3. **Chunk** → 500 chars with 100 overlap
4. **Embed** → Google Gemini text-embedding-004
5. **Store** → ChromaDB with metadata
6. **Analyze** → Gemini 2.5 Flash generates summary
7. **Save** → Summary to PostgreSQL

### Chat RAG
1. **Query** → User asks question
2. **Embed** → Embed query with Gemini
3. **Retrieve** → 
   - Top 5 chunks from user's documents (ChromaDB)
   - Top 3 chunks from medical knowledge (ChromaDB)
   - Recent record summaries (PostgreSQL)
4. **Generate** → Gemini 2.5 Flash with combined context
5. **Respond** → Answer + sources cited

## Production Deployment

### Checklist
- [ ] Use managed PostgreSQL (Supabase, AWS RDS, etc.)
- [ ] Use cloud storage (S3, GCS) instead of local files
- [ ] Add proper OAuth2 + JWT authentication
- [ ] Set up Celery for background tasks
- [ ] Use Redis for caching
- [ ] Enable HTTPS
- [ ] Add rate limiting
- [ ] Set up monitoring (Sentry, CloudWatch)
- [ ] Implement database backups
- [ ] Add error tracking
- [ ] Configure CORS for production domain

### Environment Variables (Production)
```
DATABASE_URL=postgresql://...
GOOGLE_API_KEY=...
SECRET_KEY=<generate with: openssl rand -hex 32>
ENVIRONMENT=production
SENTRY_DSN=...
ALLOWED_ORIGINS=https://yourdomain.com
```

## Testing

```bash
# Run tests
pytest

# Test coverage
pytest --cov=backend --cov-report=html

# API tests
python -m httpx http://localhost:8000/health
```

## Troubleshooting

### OCR not working
- Install Tesseract OCR (see setup instructions)
- Set `TESSERACT_PATH` in environment if not in PATH

### ChromaDB errors
- Delete `./chroma_db` and re-run ingestion
- Check disk space (embeddings can be large)

### Gemini API errors
- Verify API key is valid
- Check quota/billing in Google Cloud Console
- Add rate limiting/retry logic

### Database connection errors
- Verify PostgreSQL is running
- Check DATABASE_URL format
- Ensure database exists

## Support & Documentation

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Gemini API**: https://ai.google.dev/docs
- **ChromaDB**: https://docs.trychroma.com/
- **SQLAlchemy**: https://docs.sqlalchemy.org/

---

Built for the KEEP health platform. All medical information is for educational purposes only and not a substitute for professional medical advice.
