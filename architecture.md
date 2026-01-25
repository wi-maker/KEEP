# KEEP Platform — System Architecture

> **Version:** 1.0  
> **Last Updated:** January 2026  
> **Author:** Engineering Leadership Documentation

---

## Executive Summary

**KEEP** is an intelligent personal health vault platform enabling individuals and families to securely store, organize, and understand their medical records. The system combines a **FastAPI backend**, **Supabase authentication and storage**, **ChromaDB vector database**, and a **RAG-powered AI assistant ("Kelly")** to deliver context-aware medical document analysis and conversational support.

---

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Single-Page Application (Vanilla HTML/JS)               │    │
│  │                    TailwindCSS • Supabase JS SDK                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                           JWT Bearer Token (HTTPS)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FastAPI Application Server                        │    │
│  │         REST Endpoints • SSE Streaming • Background Tasks            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                │                │
         ┌──────────┘                │                └──────────┐
         ▼                           ▼                           ▼
┌─────────────────┐      ┌───────────────────┐      ┌─────────────────────┐
│   DATA LAYER    │      │   AI/ML LAYER     │      │  EXTERNAL SERVICES  │
│  ┌───────────┐  │      │ ┌───────────────┐ │      │ ┌─────────────────┐ │
│  │ PostgreSQL│  │      │ │   OpenRouter  │ │      │ │  Supabase Auth  │ │
│  │ (Supabase)│  │      │ │   (Primary)   │ │      │ │  & Storage      │ │
│  └───────────┘  │      │ └───────────────┘ │      │ └─────────────────┘ │
│  ┌───────────┐  │      │ ┌───────────────┐ │      │ ┌─────────────────┐ │
│  │ ChromaDB  │  │      │ │Google Gemini  │ │      │ │  Tesseract OCR  │ │
│  │ (Vectors) │  │      │ │  (Fallback)   │ │      │ │                 │ │
│  └───────────┘  │      │ └───────────────┘ │      │ └─────────────────┘ │
└─────────────────┘      └───────────────────┘      └─────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Frontend (Single-Page Application)

| Aspect | Details |
|--------|---------|
| **Architecture** | Single-file SPA (`index.html`, ~3000 lines) |
| **Framework** | Vanilla JavaScript with TailwindCSS |
| **Auth** | Supabase JS SDK for OAuth/Email authentication |
| **State** | In-memory JavaScript object (`app` namespace) |
| **Routing** | Hash-based view switching (dashboard, records, timeline, sharing, profile, upload) |

**Key Features:**
- Responsive design (desktop sidebar + mobile bottom nav)
- Profile switching for family management
- Real-time document upload with progress indicators
- SSE-based streaming chat interface
- Public share view (no authentication required)

---

### 2.2 Backend (FastAPI Application)

| Component | File | Responsibility |
|-----------|------|----------------|
| **API Server** | `main.py` | Route definitions, request handling, background tasks |
| **Authentication** | `auth.py` | Supabase JWT verification, token payload extraction |
| **Configuration** | `config.py` | Environment variables, AI provider settings |
| **Data Models** | `models.py` | SQLAlchemy ORM entities |
| **Schemas** | `schemas.py` | Pydantic request/response models |
| **RAG Pipeline** | `rag_pipeline.py` | Document processing, vector operations, AI generation |
| **AI Providers** | `ai_providers.py` | OpenRouter/Gemini abstraction with fallback |
| **Storage** | `storage.py` | Supabase Storage integration |
| **Utilities** | `utils.py` | OCR extraction, text chunking |
| **Knowledge Base** | `medical_kb_ingestion.py` | Medical knowledge base seeding |

**API Endpoint Categories:**

| Category | Endpoints | Auth Required |
|----------|-----------|---------------|
| Auth | `/auth/sync`, `/auth/me`, `/auth/data` | Yes |
| Profiles | `/profiles` (CRUD) | Yes |
| Records | `/records/upload`, `/records/{id}`, `/records/{id}/file` | Yes |
| Chat | `/chat`, `/chat/stream`, `/chat/history` | Yes |
| Sharing | `/shares` (CRUD), `/shares/public/{token}` | Mixed |
| Timeline | `/timeline` | Partial |
| Health | `/health` | No |

---

### 2.3 Data Layer

#### 2.3.1 Relational Database (PostgreSQL via Supabase / SQLite for dev)

```
┌──────────┐       ┌──────────┐       ┌──────────┐
│   User   │──1:N──│ Profile  │──1:N──│  Record  │
└──────────┘       └──────────┘       └──────────┘
     │                   │                  │
     │                   │                  ├──1:N── RecordFile
     │                   │                  ├──1:N── RecordText
     │                   │                  └──1:1── RecordAnalysis
     │                   │
     ├──1:N── Share      └──1:N── TimelineEvent
     └──1:N── ChatHistory
```

**Entity Descriptions:**

| Entity | Purpose | Key Fields |
|--------|---------|------------|
| `User` | Account holder | `id`, `email`, `full_name`, `phone` |
| `Profile` | Family member record container | `user_id`, `name`, `relation` |
| `Record` | Medical document metadata | `profile_id`, `title`, `status`, `record_date` |
| `RecordFile` | File storage reference | `record_id`, `file_path` (cloud URL) |
| `RecordText` | Extracted document text | `record_id`, `content`, `page_number` |
| `RecordAnalysis` | AI-generated insights | `summary`, `key_findings`, `doctor_questions` |
| `ChatHistory` | Conversation logs | `user_id`, `role`, `content`, `context_used` |
| `Share` | Secure share links | `token`, `record_ids`, `expires_at`, `views` |
| `TimelineEvent` | Activity audit log | `event_type`, `event_title`, `related_record_id` |

#### 2.3.2 Vector Database (ChromaDB)

| Collection | Purpose | Metadata |
|------------|---------|----------|
| `user_documents` | User-uploaded medical records | `record_id`, `profile_id`, `user_id`, `document_title` |
| `medical_knowledge` | Authoritative medical sources | `source`, `topic`, `url` |

**Embedding Model:** Google Gemini `text-embedding-004` (768 dimensions)

---

### 2.4 AI/ML Layer

#### 2.4.1 Provider Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AIProvider                        │
│  ┌───────────────┐      ┌───────────────────────┐   │
│  │ OpenRouter    │ ──▶  │ 400+ Models (GPT-4o,  │   │
│  │ (Primary)     │      │ Claude 3.5, Llama 3)  │   │
│  └───────────────┘      └───────────────────────┘   │
│           │                                          │
│           │ Fallback on failure                      │
│           ▼                                          │
│  ┌───────────────┐      ┌───────────────────────┐   │
│  │ Google Gemini │ ──▶  │ gemini-2.0-flash-001  │   │
│  │ (Fallback)    │      │                       │   │
│  └───────────────┘      └───────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
- **Singleton pattern** for provider management
- **Automatic failover** with structured logging
- **OpenAI SDK compatibility** for OpenRouter integration

#### 2.4.2 RAG Pipeline

```
Document Upload Flow:
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  Upload  │───▶│  OCR/Parse   │───▶│   Chunking   │───▶│  Embedding  │
│  (File)  │    │ (Tesseract/  │    │ (500 chars,  │    │  (Gemini)   │
│          │    │  PyPDF2)     │    │  100 overlap)│    │             │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                               │
                                                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         ChromaDB Storage                              │
│    user_documents collection (filtered by user_id + profile_id)      │
└──────────────────────────────────────────────────────────────────────┘

Chat Query Flow:
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  Query   │───▶│   Embed      │───▶│ Hybrid Search│───▶│  Context    │
│  (User)  │    │   Query      │    │ (Vector +    │    │  Assembly   │
│          │    │              │    │  Keyword)    │    │             │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                               │
                                                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      AI Generation (Kelly Persona)                    │
│   Context: User Docs + Medical KB | Tone: Warm, Professional         │
└──────────────────────────────────────────────────────────────────────┘
```

**RAG Settings:**
- Chunk size: 1000 characters
- Overlap: 200 characters
- Top-K retrieval: 5 user documents + 3 medical knowledge results

---

## 3. Folder-to-Architecture Mapping

```
KEEP/
├── backend/                    # API Layer
│   ├── main.py                 # FastAPI app, route handlers (882 lines)
│   ├── auth.py                 # JWT verification, TokenPayload class
│   ├── config.py               # Settings singleton (env vars)
│   ├── db.py                   # SQLAlchemy engine/session
│   ├── models.py               # ORM entities (9 models)
│   ├── schemas.py              # Pydantic DTOs (42 schemas)
│   ├── rag_pipeline.py         # RAG orchestration (690 lines)
│   ├── ai_providers.py         # LLM abstraction layer
│   ├── storage.py              # Supabase Storage client
│   ├── utils.py                # OCR, chunking utilities
│   ├── medical_kb_ingestion.py # Knowledge base seeder
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container build
│   └── .env.example            # Environment template
│
├── frontend/                   # Client Layer
│   └── index.html              # Complete SPA (3021 lines)
│
├── start.sh                    # Startup script
├── Procfile                    # Heroku/Railway deployment
└── README.md                   # Project documentation
```

---

## 4. Data Flow Diagrams

### 4.1 Document Upload & Processing

```
User                Frontend              Backend               Storage/AI
  │                    │                     │                      │
  │──Upload File──────▶│                     │                      │
  │                    │──POST /records/─────▶│                      │
  │                    │   upload            │                      │
  │                    │                     │──Upload to Supabase──▶│
  │                    │                     │◀──Public URL──────────│
  │                    │                     │                      │
  │                    │                     │──Background Task─────▶│
  │                    │                     │   (OCR + Embed +      │
  │                    │                     │    AI Analysis)       │
  │                    │◀──{record_id,───────│                      │
  │                    │    status}          │                      │
  │◀──Processing───────│                     │                      │
  │   Indicator        │                     │                      │
  │                    │                     │◀──Update status──────│
  │◀──Poll/Refresh────▶│                     │   to "Analyzed"      │
```

### 4.2 RAG Chat Interaction

```
User                Frontend              Backend                    AI
  │                    │                     │                        │
  │──Send Message─────▶│                     │                        │
  │                    │──POST /chat/stream──▶│                        │
  │                    │                     │──Embed Query───────────▶│
  │                    │                     │◀──Query Embedding───────│
  │                    │                     │                        │
  │                    │                     │──Search ChromaDB────────│
  │                    │                     │  (user_docs +          │
  │                    │                     │   medical_kb)          │
  │                    │                     │                        │
  │                    │                     │──Generate (w/ context)─▶│
  │                    │◀──SSE: sources──────│                        │
  │                    │◀──SSE: content──────│◀──Streamed response────│
  │                    │◀──SSE: done─────────│                        │
  │◀──Streaming Chat───│                     │                        │
```

---

## 5. AI Assistant Architecture ("Kelly")

### 5.1 Persona Design

Kelly operates as a **warm, empathetic health companion** with strict grounding rules:

| Attribute | Implementation |
|-----------|----------------|
| **Tone** | Caring, professional, reassuring |
| **Grounding** | Responses based strictly on retrieved context |
| **Attribution** | Every claim cites source documents |
| **Disclaimer** | Medical disclaimers on all health advice |
| **Personalization** | Awareness of user name, family profiles |

### 5.2 Context Assembly

```python
System Prompt Components:
├── Persona definition (Kelly identity)
├── User context (name, current profile)
├── Retrieved user documents (top-5 chunks)
├── Medical knowledge base (top-3 chunks)
├── Source attribution instructions
└── Response formatting guidelines
```

### 5.3 Knowledge Sources

| Source | Provider | Content |
|--------|----------|---------|
| User Documents | User uploads | Personal medical records |
| WHO | Curated KB | Lab test interpretation |
| CDC | Curated KB | Vaccination guidelines |
| Mayo Clinic | Curated KB | Common conditions |
| Cleveland Clinic | Curated KB | Vital sign ranges |
| MedlinePlus | Curated KB | Test explanations |
| NIH | Curated KB | Treatment guidelines |

---

## 6. Security Architecture

### 6.1 Authentication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│  Supabase   │────▶│   Backend   │
│  (JS SDK)   │     │    Auth     │     │  (FastAPI)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │ 1. Login          │                   │
       │   (OAuth/Email)   │                   │
       │──────────────────▶│                   │
       │                   │                   │
       │◀──────────────────│                   │
       │ 2. JWT Token      │                   │
       │                   │                   │
       │ 3. API Request    │                   │
       │   + Bearer Token  │                   │
       │───────────────────────────────────────▶│
       │                                        │
       │                   │◀──4. Verify JWT────│
       │                   │   (SUPABASE_JWT_   │
       │                   │    SECRET)         │
```

### 6.2 Security Controls

| Layer | Control | Implementation |
|-------|---------|----------------|
| **Transport** | TLS Encryption | HTTPS enforced |
| **Authentication** | JWT (HS256/RS256) | Supabase-issued tokens |
| **Authorization** | Resource ownership | `user_id` filtering on all queries |
| **Session** | Token expiry | 30-day access tokens |
| **Sharing** | Time-limited links | Unique tokens with expiry timestamps |
| **Data Isolation** | User-scoped queries | ChromaDB `user_id` metadata filter |
| **Dev Bypass** | Development mode | Mock user in non-production |

### 6.3 Data Access Patterns

```
User A ──▶ Profile A1 ──▶ Records (A1 only)
       ──▶ Profile A2 ──▶ Records (A2 only)
       
User B ──▶ Profile B1 ──▶ Records (B1 only)  ✗ Cannot access A's data
```

---

## 7. Deployment Architecture

### 7.1 Infrastructure Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Compute** | Railway / Render / Docker | API server hosting |
| **Auth/DB** | Supabase | PostgreSQL + Auth + Storage |
| **Vectors** | ChromaDB (persistent) | Embedding storage |
| **Files** | Supabase Storage | Document blob storage |
| **Frontend** | Vercel / Static hosting | SPA delivery |

### 7.2 Environment Configuration

```bash
# Required Environment Variables
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJxxx                    # Anon key for frontend
SUPABASE_JWT_SECRET=xxx                # For backend JWT verification
SUPABASE_SERVICE_ROLE_KEY=xxx          # Admin operations

GOOGLE_API_KEY=xxx                     # Embeddings + Gemini fallback
OPENROUTER_API_KEY=xxx                 # Primary LLM provider

DATABASE_URL=postgresql://...          # Production DB
CHROMA_PERSIST_DIR=/data/chroma        # Vector persistence
SECRET_KEY=xxx                         # Application secret
ENVIRONMENT=production                 # Switches auth behavior
```

### 7.3 Deployment Topology

```
┌────────────────────────────────────────────────────────────────────┐
│                           CDN (Vercel)                              │
│                      ┌─────────────────┐                           │
│                      │   index.html    │                           │
│                      │  (Static SPA)   │                           │
│                      └─────────────────┘                           │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Application Platform (Railway)                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Container                          │  │
│  │   ┌───────────┐  ┌───────────┐  ┌─────────────────────────┐  │  │
│  │   │ Uvicorn   │──│ FastAPI   │──│ ChromaDB (Persistent)   │  │  │
│  │   │ ASGI      │  │ App       │  │                         │  │  │
│  │   └───────────┘  └───────────┘  └─────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                    │                          │
                    ▼                          ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│         Supabase             │  │       AI Providers               │
│  ┌────────────────────────┐  │  │  ┌──────────────────────────┐   │
│  │   PostgreSQL Database  │  │  │  │       OpenRouter          │   │
│  │   Auth Service         │  │  │  │  (GPT-4o, Claude 3.5)     │   │
│  │   Object Storage       │  │  │  └──────────────────────────┘   │
│  └────────────────────────┘  │  │  ┌──────────────────────────┐   │
└──────────────────────────────┘  │  │       Google AI           │   │
                                  │  │  (Gemini + Embeddings)    │   │
                                  │  └──────────────────────────┘   │
                                  └──────────────────────────────────┘
```

---

## 8. Scalability Considerations

### 8.1 Current Architecture Constraints

| Component | Limitation | Mitigation Strategy |
|-----------|------------|---------------------|
| **SQLite** | Single-node only | PostgreSQL for production |
| **ChromaDB** | In-process, file-based | Chroma Cloud or Pinecone at scale |
| **Background Tasks** | In-process queue | Celery + Redis for multi-worker |
| **File Storage** | Supabase limits | S3 with signed URLs |
| **LLM Costs** | Per-token billing | Response caching, prompt optimization |

### 8.2 Scaling Strategies

**Horizontal Scaling:**
```
Load Balancer
      │
      ├── API Instance 1 ──▶ Shared PostgreSQL
      ├── API Instance 2 ──▶ Shared ChromaDB (external)
      └── API Instance 3 ──▶ Shared Redis (task queue)
```

**Recommended Upgrades for Scale:**

1. **Database:** Migrate to managed PostgreSQL (Supabase, RDS)
2. **Vectors:** Migrate ChromaDB to cloud-hosted (Pinecone, Weaviate)
3. **Tasks:** Implement Celery with Redis broker
4. **Caching:** Add Redis layer for chat history, embeddings
5. **Storage:** Direct S3 integration with presigned URLs

---

## 9. External Integrations

| Integration | Purpose | Protocol |
|-------------|---------|----------|
| **Supabase Auth** | User identity management | OAuth 2.0 / JWT |
| **Supabase Storage** | Document blob storage | REST API |
| **OpenRouter** | Primary LLM access | OpenAI-compatible API |
| **Google Gemini** | Fallback LLM + Embeddings | REST API |
| **Tesseract OCR** | Image text extraction | Local binary |
| **PyPDF2** | PDF text extraction | Python library |

---

## 10. Summary

KEEP demonstrates a **modern, AI-augmented health platform** architecture with:

- **Clean separation** between frontend SPA and API backend
- **Robust RAG implementation** with hybrid search (vector + keyword)
- **Production-ready auth** via Supabase JWT integration
- **Resilient AI layer** with automatic provider fallback
- **Family-centric data model** supporting multi-profile management
- **Secure sharing** with time-limited, revocable links

The architecture prioritizes **user trust and data privacy** while delivering sophisticated AI capabilities through a thoughtfully designed RAG pipeline.

---

*This document is intended for engineering leadership and technical stakeholders. For developer onboarding, see `backend/README.md`.*
