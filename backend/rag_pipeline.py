"""
RAG Pipeline for KEEP Platform
- Document processing (OCR, PDF extraction)
- ChromaDB vector storage
- OpenRouter primary + Google Gemini fallback for generation
- Google Gemini embeddings (required for ChromaDB compatibility)
- Medical knowledge base integration
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings
import google.generativeai as genai
from sqlalchemy.orm import Session

from backend import models
from backend.db import SessionLocal
from backend.config import settings
from backend.utils import extract_text_from_pdf, extract_text_from_image, chunk_text
from backend.ai_providers import get_ai_provider, AIProviderError

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google AI for embeddings (required for ChromaDB)
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    logger.info("Google AI configured for embeddings")
else:
    logger.warning("GOOGLE_API_KEY is missing. Embeddings will not work.")

class AIClient:
    """
    Unified AI client using provider abstraction with fallback.
    
    Uses OpenRouter as primary provider with Gemini as fallback.
    Configuration is managed in config.py via environment variables.
    """
    
    def __init__(self):
        self.provider = get_ai_provider()
        logger.info(f"AIClient initialized with providers: {self.provider.get_status()['active_providers']}")
        
    def generate_content(self, prompt: str) -> str:
        """Generate content with automatic fallback"""
        try:
            return self.provider.generate(prompt)
        except AIProviderError as e:
            logger.error(f"All AI providers failed: {e}")
            raise e


class EmbeddingClient:
    """
    Embedding client using Google Gemini API.
    
    Embeddings always use Google API for ChromaDB compatibility.
    The embedding model is configured in settings.EMBEDDING_MODEL.
    """

    def embed_content(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embeddings using Google Gemini"""
        try:
            model = settings.EMBEDDING_MODEL
            result = genai.embed_content(
                model=model,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise e

class ChromaDBClient:
    """Manages ChromaDB vector store"""
    
    def __init__(self):
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        
        # Collections
        self.user_docs_collection = self.client.get_or_create_collection(
            name="user_documents",
            metadata={"description": "User uploaded medical documents"}
        )
        
        self.medical_kb_collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"description": "Authoritative medical knowledge base"}
        )
    
    def add_user_document(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        record_id: str,
        profile_id: str,
        user_id: str,
        metadata: Dict
    ):
        """Add user document chunks to ChromaDB"""
        ids = [f"{record_id}_chunk_{i}" for i in range(len(chunks))]
        
        metadatas = [{
            "record_id": record_id,
            "profile_id": profile_id,
            "user_id": user_id,
            "chunk_index": i,
            "source_type": "user_document",
            **metadata
        } for i in range(len(chunks))]
        
        self.user_docs_collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def delete_user_document(self, record_id: str):
        """Delete all chunks for a record"""
        self.user_docs_collection.delete(
            where={"record_id": record_id}
        )
    
    def query_user_documents(
        self,
        query_embedding: List[float],
        user_id: str,
        query_text: str = None,
        profile_id: str = None,
        n_results: int = 5
    ) -> Dict:
        """Query user's documents with hybrid search (vector + keyword).
        
        Args:
            query_embedding: Vector embedding for semantic search
            user_id: User ID filter
            query_text: Optional text for BM25 keyword search (hybrid mode)
            profile_id: Optional profile filter
            n_results: Number of results to return
        """
        where_filter = {"user_id": user_id}
        if profile_id:
            where_filter["profile_id"] = profile_id
        
        try:
            # Build query parameters for hybrid search
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "where": where_filter,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add keyword search for hybrid mode
            if query_text:
                query_params["query_texts"] = [query_text]
                logger.debug(f"Hybrid search: vector + keyword for '{query_text[:50]}...'")
            
            results = self.user_docs_collection.query(**query_params)
            return results
        except Exception as e:
            logger.error(f"Error querying user documents: {e}")
            # Return empty result structure
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
    
    def query_medical_knowledge(
        self,
        query_embedding: List[float],
        query_text: str = None,
        n_results: int = 3
    ) -> Dict:
        """Query medical knowledge base with hybrid search (vector + keyword).
        
        Args:
            query_embedding: Vector embedding for semantic search
            query_text: Optional text for BM25 keyword search (hybrid mode)
            n_results: Number of results to return
        """
        try:
            # Build query parameters for hybrid search
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add keyword search for hybrid mode
            if query_text:
                query_params["query_texts"] = [query_text]
            
            results = self.medical_kb_collection.query(**query_params)
            return results
        except Exception as e:
            logger.warning(f"Error querying medical knowledge: {e}")
            # Return empty result structure
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }


class EmbeddingService:
    """Handles text embedding with Google Gemini API"""
    
    def __init__(self):
        self.client = EmbeddingClient()

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.client.embed_content(text, "retrieval_document")
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        return self.client.embed_content(query, "retrieval_query")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings


class DocumentProcessor:
    """Processes medical documents using AI provider abstraction"""
    
    def __init__(self, db: Session, chroma_client: ChromaDBClient):
        self.db = db
        self.chroma = chroma_client
        self.ai_client = AIClient()  # Uses OpenRouter primary + Gemini fallback
        self.embedding_service = EmbeddingService()
    
    def process_document(
        self,
        record_id: str,
        file_path: str,
        file_type: str
    ):
        """Main document processing pipeline"""
        try:
            # 1. Extract text
            logger.info(f"Step 1 - Extracting text for {record_id}")
            file_type = file_type.lower()
            
            if file_type == 'pdf':
                text = extract_text_from_pdf(file_path)
            elif file_type in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
                text = extract_text_from_image(file_path)
            elif file_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_type in ['doc', 'docx']:
                # Handle Word documents
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    raise ValueError("python-docx package required for DOCX files. Install with: pip install python-docx")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Extracted text length: {len(text)}")

            # 2. Save extracted text
            logger.info("Step 2 - Saving text")
            record = self.db.query(models.Record).filter_by(id=record_id).first()
            if not record:
                raise ValueError(f"Record {record_id} not found")

            record_text = models.RecordText(
                record_id=record_id,
                content=text,
                page_number=1
            )
            self.db.add(record_text)
            self.db.commit()
            
            # 3. Chunk text
            logger.info("Step 3 - Chunking text")
            chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            logger.info(f"Generated {len(chunks)} chunks")
            
            # 4. Generate embeddings
            logger.info("Step 4 - Generating embeddings")
            embeddings = self.embedding_service.embed_batch(chunks)
            logger.info("Embeddings generated")
            
            # 5. Store in ChromaDB
            logger.info("Step 5 - Storing in ChromaDB")
            self.chroma.add_user_document(
                chunks=chunks,
                embeddings=embeddings,
                record_id=record_id,
                profile_id=record.profile_id,
                user_id=record.profile.user_id,
                metadata={
                    "title": record.title,
                    "record_type": record.record_type or "Unknown",
                    "date": str(record.record_date) if record.record_date else ""
                }
            )
            logger.info("Stored in ChromaDB")
            
            # 6. Generate AI analysis
            logger.info("Step 6 - Generating analysis")
            analysis = self.generate_analysis(text, record_id)
            logger.info("Analysis generated")
            
            # 7. Update record status
            logger.info("Step 7 - Updating status")
            record.status = "Analyzed"
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            # Update record with error
            try:
                record = self.db.query(models.Record).filter_by(id=record_id).first()
                if record:
                    record.status = "Error"
                    
                    # For OCR errors, create a helpful fallback analysis
                    if "tesseract" in str(e).lower() or "ocr" in str(e).lower():
                        # Create a fallback analysis explaining the OCR issue
                        analysis = models.RecordAnalysis(
                            record_id=record_id,
                            summary="Unable to extract text from this document automatically. Please ensure Tesseract OCR is installed for image processing.",
                            key_findings=[
                                "OCR text extraction failed",
                                "File uploaded successfully but text could not be read",
                                "Manual review of the document is recommended"
                            ],
                            confidence_score=0.0,
                            doctor_questions=[
                                "Can you provide a text-based version of this document?",
                                "Is there an alternative format available (e.g., searchable PDF)?",
                                "What are the key points I should know from this document?"
                            ],
                            medical_sources_used=[]
                        )
                        self.db.add(analysis)
                    
                    self.db.commit()
            except Exception as db_e:
                logger.error(f"Failed to update record status to Error: {db_e}")
            raise e
    
    def generate_analysis(self, document_text: str, record_id: str) -> models.RecordAnalysis:
        """Generate AI analysis of medical document"""
        
        # Retrieve relevant medical knowledge
        doc_embedding = self.embedding_service.embed_text(document_text[:1000])  # First 1000 chars
        medical_context = self.chroma.query_medical_knowledge(doc_embedding, n_results=3)
        
        medical_knowledge_text = "\n\n".join([
            f"Source: {meta.get('source_name', 'Unknown')}\n{doc}"
            for doc, meta in zip(
                medical_context.get('documents', [[]])[0],
                medical_context.get('metadatas', [[]])[0]
            )
        ])
        
        prompt = f"""You are a medical AI assistant that provides clear, professional health document analysis.

DOCUMENT TEXT:
{document_text[:3000]}

RELEVANT MEDICAL CONTEXT:
{medical_knowledge_text}

Your task is to analyze this medical document and provide a comprehensive, easy-to-understand explanation.

IMPORTANT INSTRUCTIONS:
- Do NOT address the user by any name (real or assumed)
- Start with: "Based on the medical document you uploaded, here is a clear explanation:"
- Provide detailed, medically robust information that remains accessible to non-medical readers
- Use clear, simple sentences broken down logically
- Structure your response as: Summary → Key Details → What It Means → Recommendations

Provide the following analysis in JSON format:

1. SUMMARY: A detailed professional summary (3-4 sentences) explaining what this document contains and its medical purpose. Start with "Based on the medical document you uploaded, here is a clear explanation:" followed by the medical context.

2. KEY FINDINGS: List 4-6 specific, detailed observations or results from the document. Each finding should be medically accurate but explained in simple terms. Include relevant numbers, measurements, or test results.

3. MEDICAL SIGNIFICANCE: Explain what these findings mean in practical terms. What should the patient understand about their health based on these results?

4. CONFIDENCE SCORE: 0-100 (how clearly and completely you can interpret this document based on available information)

5. QUESTIONS FOR DOCTOR: 3-4 specific, relevant questions the patient should ask their healthcare provider about these results

6. MEDICAL SOURCES: List the reputable medical sources used (WHO, Mayo Clinic, CDC, NIH, NHS UK, MedlinePlus, Cleveland Clinic, Johns Hopkins Medicine, American Heart Association, American Cancer Society, etc.)

IMPORTANT: Return ONLY valid JSON in this exact format:
{{
  "summary": "Based on the medical document you uploaded, here is a clear explanation: [Your detailed summary here]",
  "findings": ["Detailed finding 1...", "Detailed finding 2...", "Detailed finding 3...", "Detailed finding 4..."],
  "medical_significance": "What these findings mean for the patient's health in simple terms...",
  "confidence": 95,
  "questions": ["Specific question 1?", "Specific question 2?", "Specific question 3?"],
  "sources": ["Source Name 1", "Source Name 2", "Source Name 3"]
}}

Do not include any text before or after the JSON."""

        response_text = self.ai_client.generate_content(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            analysis_data = json.loads(response_text.strip())
            
            # Extract sources from the response, or use medical_significance as additional context
            sources_list = analysis_data.get('sources', [])
            
            # Create analysis record
            analysis = models.RecordAnalysis(
                record_id=record_id,
                summary=analysis_data.get('summary', ''),
                key_findings=analysis_data.get('findings', []),
                confidence_score=float(analysis_data.get('confidence', 0)),
                doctor_questions=analysis_data.get('questions', []),
                medical_sources_used=sources_list
            )
            
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}. Response: {response_text}")
            # Fallback: create basic analysis
            analysis = models.RecordAnalysis(
                record_id=record_id,
                summary="Document analyzed. Please consult with your healthcare provider for interpretation.",
                key_findings=["Analysis completed"],
                confidence_score=50.0,
                doctor_questions=["What do these results mean for my health?"],
                medical_sources_used=[]
            )
            self.db.add(analysis)
            self.db.commit()
            return analysis


class RAGChatbot:
    """RAG-powered chatbot for medical Q&A using AI provider abstraction"""
    
    def __init__(self, db: Session, chroma_client: ChromaDBClient):
        self.db = db
        self.chroma = chroma_client
        self.ai_client = AIClient()  # Uses OpenRouter primary + Gemini fallback
        self.embedding_service = EmbeddingService()
    
    def chat(
        self,
        user_id: str,
        profile_id: str,
        message: str,
        user_name: str = "User"
    ) -> Tuple[str, List[Dict]]:
        """Process chat message with RAG"""
        
        # 1. Embed query
        query_embedding = self.embedding_service.embed_query(message)
        
        # 2. Retrieve user documents (HYBRID SEARCH: vector + keyword)
        user_docs = self.chroma.query_user_documents(
            query_embedding,
            user_id=user_id,
            query_text=message,  # Enable keyword search
            profile_id=profile_id,
            n_results=5
        )
        
        # 3. Retrieve medical knowledge (HYBRID SEARCH: vector + keyword)
        medical_kb = self.chroma.query_medical_knowledge(
            query_embedding,
            query_text=message,  # Enable keyword search
            n_results=3
        )
        
        # 4. Get conversation history for context (last 10 messages)
        chat_history = self.db.query(models.ChatHistory)\
            .filter_by(user_id=user_id, profile_id=profile_id)\
            .order_by(models.ChatHistory.timestamp.desc())\
            .limit(10)\
            .all()
        
        # Format conversation history (reverse to chronological order)
        conversation_history = ""
        if chat_history:
            history_items = []
            for h in reversed(chat_history):
                role_label = "User" if h.role == "user" else "Assistant"
                # Truncate long messages to keep prompt manageable
                content = h.content[:500] + "..." if len(h.content) > 500 else h.content
                history_items.append(f"{role_label}: {content}")
            conversation_history = "\n".join(history_items)
        
        # 5. Get user's record summaries
        profile = self.db.query(models.Profile).filter_by(id=profile_id).first()
        records = self.db.query(models.Record)\
            .filter_by(profile_id=profile_id, status="Analyzed")\
            .order_by(models.Record.created_at.desc())\
            .limit(5)\
            .all()
        
        summaries_text = "\n\n".join([
            f"Record: {r.title} ({r.record_date.strftime('%Y-%m-%d') if r.record_date else 'N/A'})\nSummary: {r.analysis.summary if r.analysis else 'No analysis'}"
            for r in records
        ])
        
        # 6. Build context
        user_context = "\n\n".join([
            f"From {meta.get('title', 'Unknown')}: {doc}"
            for doc, meta in zip(
                user_docs.get('documents', [[]])[0],
                user_docs.get('metadatas', [[]])[0]
            )
        ])
        
        medical_context = "\n\n".join([
            f"Medical Source ({meta.get('source_name', 'Unknown')}): {doc}"
            for doc, meta in zip(
                medical_kb.get('documents', [[]])[0],
                medical_kb.get('metadatas', [[]])[0]
            )
        ])
        
        # 7. Generate response with conversation history
        prompt = f"""You are 'Kelly', a professional yet warm and empathetic medical AI assistant for the KEEP platform.
You are speaking directly to {user_name}.

YOUR ROLE:
- Help {user_name} understand their health records in simple, human terms.
- Be calm, reassuring, and professional.
- Use natural conversation (e.g., "I noticed in your report...", "Does that make sense?").
- If the user asks a follow-up question (e.g., "Really?", "Why?"), answer it naturally based on the previous context.

USER CONTEXT:
- Name: {user_name}
- Profile: {profile.name} ({profile.relation})

DOCUMENT OWNERSHIP LOGIC:
- Look at the "RELEVANT DOCUMENT EXCERPTS" below.
- If you see the name "{user_name}" or "{profile.name}" in the document text, assume this is THEIR record. Address them directly (e.g., "Your blood test shows...").
- If the name in the document is different, refer to "the patient" or "the report".

RELEVANT DOCUMENT EXCERPTS:
{user_context}

MEDICAL KNOWLEDGE:
{medical_context}

USER'S RECENT RECORDS SUMMARY:
{summaries_text}

CONVERSATION HISTORY:
{conversation_history if conversation_history else "No previous conversation."}

CURRENT MESSAGE from {user_name}:
{message}

INSTRUCTIONS:
1. Answer {user_name}'s question using the provided context.
2. If the user asks about specific results, explain them simply and check if they fall within normal ranges based on the Medical Knowledge provided.
3. Be honest if you don't know. Say "I don't see that in the documents provided" rather than guessing.
4. IMPORTANT: End with a standard medical disclaimer, but keep it brief and natural (e.g., "Remember, I'm an AI, so please check with your doctor for advice.").
"""

        response_text = self.ai_client.generate_content(prompt)
        
        # 7. Extract sources
        sources = []
        
        # Add user document sources
        for meta in user_docs.get('metadatas', [[]])[0]:
            sources.append({
                "type": "user_document",
                "title": meta.get('title', 'Unknown Record'),
                "excerpt": None
            })
        
        # Add medical knowledge sources
        for meta in medical_kb.get('metadatas', [[]])[0]:
            sources.append({
                "type": "medical_knowledge",
                "title": meta.get('source_name', 'Medical Reference'),
                "excerpt": meta.get('url', None)
            })
        
        # Remove duplicates
        unique_sources = []
        seen = set()
        for source in sources:
            key = (source['type'], source['title'])
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        
        return response_text, unique_sources[:5]  # Limit to 5 sources


# Singleton instances
chroma_client = ChromaDBClient()
embedding_service = EmbeddingService()


def process_record(record_id: str, file_path: str, file_type: str):
    """Process a single record (called from API)"""
    logger.info(f"Starting processing for record {record_id}, file: {file_path}")
    db = SessionLocal()
    try:
        processor = DocumentProcessor(db, chroma_client)
        processor.process_document(record_id, file_path, file_type)
        
        # Add Timeline Event for Analysis Completion
        record = db.query(models.Record).filter_by(id=record_id).first()
        if record:
            event = models.TimelineEvent(
                user_id=record.profile.user_id,
                profile_id=record.profile_id,
                event_type="analysis",
                event_title=f"AI Analysis Complete: {record.title}"
            )
            db.add(event)
            db.commit()
            
        logger.info(f"Finished processing record {record_id}")
    except Exception as e:
        error_msg = f"Failed processing record {record_id}: {e}"
        logger.error(error_msg)
        
        # Update record status to Error in DB
        try:
            # Re-open session to update status if processor didn't
            db_err = SessionLocal()
            record = db_err.query(models.Record).filter_by(id=record_id).first()
            if record:
                record.status = "Error"
                db_err.commit()
            db_err.close()
        except Exception as db_e:
            logger.error(f"Failed to update record status in fallback: {db_e}")
    finally:
        db.close()


def chat_with_rag(db: Session, user_id: str, profile_id: str, message: str, user_name: str = "User") -> Tuple[str, List[Dict]]:
    """Chat with RAG (called from API)"""
    chatbot = RAGChatbot(db, chroma_client)
    return chatbot.chat(user_id, profile_id, message, user_name=user_name)
