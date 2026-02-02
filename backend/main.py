"""
KEEP API - FastAPI Backend
Complete implementation matching frontend features
"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import os
import io
import uuid
import json
import asyncio
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote, urlparse

import models
import schemas
import db
from rag_pipeline import process_record, chat_with_rag
from config import settings
from utils import get_file_extension, sanitize_filename, get_file_size
from auth import verify_token, TokenPayload, get_current_user_id
from storage import upload_file_to_supabase, download_file_from_supabase
from routers import admin as admin_router
from routers import user as user_router

# Create tables
models.Base.metadata.create_all(bind=db.engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",          # Local development
        "https://keep-ten-pi.vercel.app/",
        "https://keep-admin.vercel.app/",
        "https://app.onkeep.co/", # <-- REPLACE THIS with your actual Vercel URL                             # Keep "*" only if you want to allow ANY website to connect (less secure)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include admin router
app.include_router(admin_router.router, prefix=settings.API_V1_STR)
app.include_router(user_router.router, prefix=settings.API_V1_STR)

# Dependency
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "KEEP API is running. Please access via the Vercel frontend.",
        "docs": "/docs"
    }

# ============================================================================
# AUTH ROUTES (Simplified - extend with proper JWT in production)
# ============================================================================

# ============================================================================
# AUTH ROUTES (Production: Supabase JWT)
# ============================================================================

@app.post(f"{settings.API_V1_STR}/auth/sync", response_model=schemas.User)
def sync_user(
    user_token: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Sync Supabase user to local database.
    Called by frontend after Supabase login.
    """
    user_id = user_token.user_id
    email = user_token.email
    full_name = user_token.full_name or email.split("@")[0]
    phone = user_token.phone  # Extract phone from token
    
    # Check if user exists
    user = db.query(models.User).filter_by(id=user_id).first()
    
    if not user:
        # Create new user
        user = models.User(
            id=user_id,  # Use Supabase ID
            email=email,
            full_name=full_name,
            phone=phone,
            is_active=True
        )
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
            
            # Create default 'Me' profile
            default_profile = models.Profile(
                user_id=user.id,
                name=full_name,
                relation="Me"
            )
            db.add(default_profile)
            db.commit()
            
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Update existing user with latest data from token
        user.full_name = full_name
        user.phone = phone
        db.commit()
        db.refresh(user)
            
    return user

@app.get(f"{settings.API_V1_STR}/auth/me", response_model=schemas.User)
def get_current_user_profile(
    user_token: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get current user (verified by token)"""
    user = db.query(models.User).filter_by(id=user_token.user_id).first()
    if not user:
        # If user has token but not in DB, auto-sync
        return sync_user(user_token, db)
    return user


@app.delete(f"{settings.API_V1_STR}/auth/data")
def delete_user_medical_data(
    user_token: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Delete all medical data for the current user.
    This deletes Records, Files, TimelineEvents, and Vector Embeddings.
    The User and Profile remain active.
    """
    user_id = user_token.user_id
    
    # Find the user
    user = db.query(models.User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Import chroma_client for vector deletion
    from rag_pipeline import chroma_client
    import logging
    
    deleted_records = 0
    deleted_events = 0
    
    # Delete vector data and records for all user's profiles
    for profile in user.profiles:
        # Delete timeline events for this profile
        events = db.query(models.TimelineEvent).filter_by(profile_id=profile.id).all()
        for event in events:
            db.delete(event)
            deleted_events += 1
        
        # Delete records and their vector embeddings
        for record in profile.records:
            try:
                chroma_client.delete_user_document(record.id)
            except Exception as e:
                logging.warning(f"Failed to delete vector data for record {record.id}: {e}")
            
            # Files are deleted via cascade when record is deleted
            db.delete(record)
            deleted_records += 1
    
    # Also delete user-level timeline events (those without profile_id)
    user_events = db.query(models.TimelineEvent).filter_by(user_id=user_id, profile_id=None).all()
    for event in user_events:
        db.delete(event)
        deleted_events += 1
    
    # Clear chat history
    db.query(models.ChatHistory).filter_by(user_id=user_id).delete()
    
    # Delete shares
    db.query(models.Share).filter_by(user_id=user_id).delete()
    
    db.commit()
    
    return {
        "message": "Medical data deleted successfully",
        "deleted_records": deleted_records,
        "deleted_events": deleted_events
    }


@app.delete(f"{settings.API_V1_STR}/auth/me")
def delete_current_user_account(
    user_token: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Permanently delete the current user's account.
    This will delete all user data including profiles, records, vector embeddings,
    and also removes the user from Supabase Auth.
    """
    user_id = user_token.user_id
    
    # Find the user
    user = db.query(models.User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Import chroma_client for vector deletion
    from rag_pipeline import chroma_client
    import logging
    
    # Delete vector data for all user's records
    for profile in user.profiles:
        for record in profile.records:
            try:
                chroma_client.delete_user_document(record.id)
            except Exception as e:
                logging.warning(f"Failed to delete vector data for record {record.id}: {e}")
    
    # Delete the user from local DB (cascade will handle profiles, records, shares, chat_history)
    db.delete(user)
    db.commit()
    
    # Delete user from Supabase Auth (requires service role key)
    try:
        from storage import supabase
        # Use admin API to delete authäuser
        supabase.auth.admin.delete_user(user_id)
    except Exception as e:
        import logging
        logging.warning(f"Failed to delete user from Supabase Auth: {e}")
        # Continue anyway - local data is already deleted
    
    return {"message": "Account successfully deleted"}


@app.patch(f"{settings.API_V1_STR}/users/welcome-modal")
def dismiss_welcome_modal(
    user_token: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Mark welcome modal as seen for current user."""
    user = db.query(models.User).filter_by(id=user_token.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.has_seen_welcome_modal = True
    db.commit()
    
    return {"status": "success", "message": "Welcome modal dismissed"}


# ============================================================================
# PROFILE ROUTES
# ============================================================================

@app.post(f"{settings.API_V1_STR}/profiles", response_model=schemas.Profile)
def create_profile(
    profile: schemas.ProfileCreate,
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Create new family profile"""
    user_id = user.user_id
    db_profile = models.Profile(
        user_id=user_id,
        name=profile.name,
        relation=profile.relation,
        avatar_seed=profile.avatar_seed
    )
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    # Create timeline event
    event = models.TimelineEvent(
        user_id=user_id,
        profile_id=db_profile.id,
        event_type="profile_created",
        event_title=f"Profile created for {profile.name}"
    )
    db.add(event)
    db.commit()
    
    return db_profile

@app.get(f"{settings.API_V1_STR}/profiles", response_model=List[schemas.Profile])
def get_profiles(
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get all profiles for user"""
    return db.query(models.Profile).filter(models.Profile.user_id == user.user_id).all()

@app.get(f"{settings.API_V1_STR}/profiles/{{profile_id}}", response_model=schemas.Profile)
def get_profile(profile_id: str, db: Session = Depends(get_db)):
    """Get profile by ID"""
    profile = db.query(models.Profile).filter_by(id=profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

# ============================================================================
# RECORD ROUTES
# ============================================================================


@app.post(f"{settings.API_V1_STR}/records/upload", response_model=schemas.UploadResponse)
async def upload_record(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    profile_id: str = Form(...),
    document_name: str = Form(...),
    record_date: Optional[str] = Form(None),
    doctor_name: Optional[str] = Form(None),
    facility_name: Optional[str] = Form(None),
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Upload medical document to Cloud Storage and process with AI"""
    
    # Validate profile and ownership
    profile = db.query(models.Profile).filter_by(id=profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    if profile.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this profile")
    
    # Create record
    db_record = models.Record(
        profile_id=profile_id,
        title=document_name,
        record_date=datetime.fromisoformat(record_date) if record_date else datetime.now(),
        doctor_name=doctor_name,
        facility_name=facility_name,
        status="Processing"
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    # --- CHANGED: Upload to Supabase Storage ---
    file_extension = get_file_extension(file.filename)
    safe_filename = sanitize_filename(file.filename)
    
    # Create a unique path: user_id/record_id/filename
    storage_path = f"{user.user_id}/{db_record.id}/{safe_filename}"
    
    try:
        public_url = await upload_file_to_supabase(file, storage_path)
    except Exception as e:
        db.delete(db_record)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to upload file to cloud: {str(e)}")

    # NOTE: We still save to a temp local file for the AI processing step
    # because many AI libraries need a physical file path.
    # We will clean this up later or let the OS handle temp dir cleanup.
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    temp_file_path = os.path.join(settings.UPLOAD_DIR, f"{db_record.id}_{safe_filename}")
    await file.seek(0)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create file record in DB (Storing Cloud URL)
    db_file = models.RecordFile(
        record_id=db_record.id,
        file_path=public_url,  # Store Cloud URL here!
        file_type=file_extension,
        original_filename=file.filename,
        file_size=get_file_size(temp_file_path)
    )
    db.add(db_file)
    db.commit()
    
    # Create timeline event
    event = models.TimelineEvent(
        user_id=profile.user_id,
        profile_id=profile_id,
        event_type="upload",
        event_title=f"Uploaded {document_name}",
        related_record_id=db_record.id
    )
    db.add(event)
    db.commit()
    
    # Process document in background (Pass the TEMP local path for processing)
    background_tasks.add_task(
        process_record,
        db_record.id,
        temp_file_path, # Local temp path for AI
        file_extension
    )
    
    return {
        "record_id": db_record.id,
        "status": "Processing",
        "message": "Document uploaded successfully. AI analysis in progress."
    }

@app.get(f"{settings.API_V1_STR}/records", response_model=List[schemas.Record])
def get_records(
    profile_id: str,
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get all records for a profile"""
    # Verify profile ownership
    profile = db.query(models.Profile).filter_by(id=profile_id).first()
    if not profile or profile.user_id != user.user_id:
        # In production, return empty list or 404 to avoid leaking existence
        return []
        
    records = db.query(models.Record)\
        .filter(models.Record.profile_id == profile_id)\
        .order_by(models.Record.created_at.desc())\
        .all()
    return records

@app.get(f"{settings.API_V1_STR}/records/{{record_id}}", response_model=schemas.Record)
def get_record(record_id: str, db: Session = Depends(get_db)):
    """Get record detail with file and AI analysis"""
    record = db.query(models.Record)\
        .filter(models.Record.id == record_id)\
        .first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return record

@app.get(f"{settings.API_V1_STR}/records/{{record_id}}/file")
def get_file(
    record_id: str,
    inline: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get record file - Streams file content directly from backend.
    Hides the underlying Supabase storage URL from users.
    """
    record = db.query(models.Record).filter_by(id=record_id).first()
    
    # 1. Check if record and file exist in DB
    if not record or not record.files:
        raise HTTPException(status_code=404, detail="File record not found")
    
    file_record = record.files[0]
    stored_url = file_record.file_path
    
    # 2. Extract relative storage path from the full Supabase URL
    # URL format: https://{project}.supabase.co/storage/v1/object/public/records/{user_id}/{record_id}/{filename}
    parsed = urlparse(stored_url)
    path_parts = parsed.path.split('/object/public/records/')
    
    if len(path_parts) < 2:
        raise HTTPException(status_code=500, detail="Invalid storage path format")
    
    # Decode URL-encoded characters (e.g., %20 -> space)
    relative_path = unquote(path_parts[1])
    
    # 3. Download file from Supabase
    file_bytes = download_file_from_supabase(relative_path)
    
    if file_bytes is None:
        raise HTTPException(status_code=404, detail="File not found in storage")
    
    # 4. Determine media type based on file extension
    extension = file_record.file_type.lower() if file_record.file_type else ''
    media_type_map = {
        'pdf': 'application/pdf',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
    }
    media_type = media_type_map.get(extension, 'application/octet-stream')
    
    # 5. Set Content-Disposition header
    disposition = 'inline' if inline else 'attachment'
    filename = file_record.original_filename or f"file.{extension}"
    
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"'
        }
    )

@app.put(f"{settings.API_V1_STR}/records/{{record_id}}", response_model=schemas.Record)
def update_record(
    record_id: str,
    update_data: schemas.RecordUpdate,
    db: Session = Depends(get_db)
):
    """Update record metadata"""
    record = db.query(models.Record).filter_by(id=record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    update_dict = update_data.dict(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(record, key, value)
    
    db.commit()
    db.refresh(record)
    return record

@app.delete(f"{settings.API_V1_STR}/records/{{record_id}}")
def delete_record(record_id: str, db: Session = Depends(get_db)):
    """Delete record and associated files"""
    record = db.query(models.Record).filter_by(id=record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Capture record details for timeline event before deletion
    record_title = record.title
    profile_id = record.profile_id
    user_id = record.profile.user_id
    
    # Delete files from disk
    for file_record in record.files:
        if os.path.exists(file_record.file_path):
            os.remove(file_record.file_path)
            # Also try to remove the directory if it's empty
            try:
                record_dir = os.path.dirname(file_record.file_path)
                if os.path.exists(record_dir) and not os.listdir(record_dir):
                    os.rmdir(record_dir)
            except Exception:
                pass  # Directory not empty or other error, ignore
    
    # Delete from ChromaDB
    from rag_pipeline import chroma_client
    chroma_client.delete_user_document(record_id)
    
    # Create timeline event before deletion
    event = models.TimelineEvent(
        user_id=user_id,
        profile_id=profile_id,
        event_type="record_deleted",
        event_title=f"Deleted {record_title}"
    )
    db.add(event)
    db.commit()
    
    # Delete from database (cascade will handle related records)
    db.delete(record)
    db.commit()
    
    return {"message": "Record deleted successfully"}

# ============================================================================
# CHAT ROUTES
# ============================================================================

@app.post(f"{settings.API_V1_STR}/chat", response_model=schemas.ChatResponse)
@app.post(f"{settings.API_V1_STR}/chat", response_model=schemas.ChatResponse)
def chat(
    request: schemas.ChatRequest,
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Chat with RAG-powered AI assistant"""
    user_id = user.user_id
    
    # Save user message
    user_msg = models.ChatHistory(
        user_id=user_id,
        profile_id=request.profile_id,
        role="user",
        content=request.message
    )
    db.add(user_msg)
    db.commit()
    
    # Get RAG response
    # Use user's name from token metadata, fallback to "User"
    user_name = user.full_name or "User"
    
    response_text, sources = chat_with_rag(
        db,
        user_id,
        request.profile_id,
        request.message,
        user_name=user_name
    )
    
    # Save assistant message
    assistant_msg = models.ChatHistory(
        user_id=user_id,
        profile_id=request.profile_id,
        role="assistant",
        content=response_text,
        context_used={"sources": sources}
    )
    db.add(assistant_msg)
    db.commit()
    
    return {
        "message": response_text,
        "sources": sources,
        "confidence": 85.0  # Can be calculated based on retrieval scores
    }

@app.post(f"{settings.API_V1_STR}/chat/stream")
@app.post(f"{settings.API_V1_STR}/chat/stream")
async def chat_stream(
    request: schemas.ChatRequest,
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Streaming chat with RAG-powered AI assistant using Server-Sent Events (SSE)"""
    user_id = user.user_id
    
    # Save user message
    user_msg = models.ChatHistory(
        user_id=user_id,
        profile_id=request.profile_id,
        role="user",
        content=request.message
    )
    db.add(user_msg)
    db.commit()
    
    async def generate_stream():
        """Generator that yields SSE events"""
        try:
            # Get RAG response (non-streaming for now, we simulate streaming on frontend)
            response_text, sources = chat_with_rag(
                db,
                user_id,
                request.profile_id,
                request.message
            )
            
            # Serialize sources for SSE (handle both dict and object types)
            serialized_sources = []
            for s in sources:
                if isinstance(s, dict):
                    serialized_sources.append({
                        'type': s.get('type', 'unknown'),
                        'title': s.get('title', 'Unknown')
                    })
                else:
                    # Handle Pydantic models or objects with attributes
                    serialized_sources.append({
                        'type': getattr(s, 'type', 'unknown'),
                        'title': getattr(s, 'title', 'Unknown')
                    })
            
            # Send sources first
            yield f"data: {json.dumps({'type': 'sources', 'data': serialized_sources})}\n\n"
            
            # Stream the response character by character (simulated streaming)
            # Break into chunks for realistic streaming effect
            chunk_size = 3  # Characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
                await asyncio.sleep(0.02)  # Small delay for streaming effect
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'data': None})}\n\n"
            
            # Save assistant message after streaming completes
            assistant_msg = models.ChatHistory(
                user_id=user_id,
                profile_id=request.profile_id,
                role="assistant",
                content=response_text,
                context_used={"sources": serialized_sources}  # Use serialized sources
            )
            db.add(assistant_msg)
            db.commit()
            
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}: {traceback.format_exc()}"
            print(f"Chat stream error: {error_detail}")  # Log to console
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get(f"{settings.API_V1_STR}/chat/history", response_model=List[schemas.ChatHistoryItem])
@app.get(f"{settings.API_V1_STR}/chat/history", response_model=List[schemas.ChatHistoryItem])
def get_chat_history(
    user: TokenPayload = Depends(verify_token),
    profile_id: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get chat history"""
    user_id = user.user_id
    query = db.query(models.ChatHistory).filter_by(user_id=user_id)
    
    if profile_id:
        query = query.filter_by(profile_id=profile_id)
    
    history = query.order_by(models.ChatHistory.timestamp.desc()).limit(limit).all()
    return reversed(history)

@app.delete(f"{settings.API_V1_STR}/chat/history")
@app.delete(f"{settings.API_V1_STR}/chat/history")
def clear_chat_history(
    user: TokenPayload = Depends(verify_token),
    profile_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Clear chat history"""
    user_id = user.user_id
    query = db.query(models.ChatHistory).filter_by(user_id=user_id)
    
    if profile_id:
        query = query.filter_by(profile_id=profile_id)
    
    query.delete()
    db.commit()
    
    return {"message": "Chat history cleared"}

# ============================================================================
# SHARING ROUTES
# ============================================================================

@app.post(f"{settings.API_V1_STR}/shares", response_model=schemas.ShareResponse)
@app.post(f"{settings.API_V1_STR}/shares", response_model=schemas.ShareResponse)
def create_share(
    share_data: schemas.ShareCreate,
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Create secure share link"""
    user_id = user.user_id
    
    # Generate unique token
    token = str(uuid.uuid4())
    
    # Calculate expiry
    expires_at = datetime.now(timezone.utc) + timedelta(hours=share_data.expiry_hours)
    
    # Create share
    db_share = models.Share(
        user_id=user_id,
        name=share_data.name,
        recipient_name=share_data.recipient_name,
        record_ids=share_data.record_ids,
        token=token,
        expires_at=expires_at,
        status="active"
    )
    db.add(db_share)
    db.commit()
    db.refresh(db_share)
    
    # Create timeline event
    event = models.TimelineEvent(
        user_id=user_id,
        profile_id=None,  # Share isn't profile-specific
        event_type="share",
        event_title=f"Shared {len(share_data.record_ids)} records with {share_data.recipient_name}"
    )
    db.add(event)
    db.commit()
    
    return schemas.ShareResponse(
        id=db_share.id,
        name=db_share.name,
        recipient_name=db_share.recipient_name,
        record_ids=db_share.record_ids,
        token=db_share.token,
        share_url=f"/s/{token}",
        expires_at=db_share.expires_at,
        status=db_share.status,
        views=db_share.views,
        created_at=db_share.created_at
    )

@app.get(f"{settings.API_V1_STR}/shares", response_model=List[schemas.ShareList])
@app.get(f"{settings.API_V1_STR}/shares", response_model=List[schemas.ShareList])
def get_shares(user: TokenPayload = Depends(verify_token), db: Session = Depends(get_db)):
    """Get all shares for user"""
    user_id = user.user_id
    shares = db.query(models.Share)\
        .filter(models.Share.user_id == user_id)\
        .order_by(models.Share.created_at.desc())\
        .all()
    
    return [{
        "id": s.id,
        "name": s.name,
        "recipient_name": s.recipient_name,
        "record_count": len(s.record_ids) if s.record_ids else 0,
        "status": s.status,
        "expires_at": s.expires_at,
        "views": s.views,
        "token": s.token
    } for s in shares]

@app.put(f"{settings.API_V1_STR}/shares/{{share_id}}/revoke")
def revoke_share(share_id: str, user: TokenPayload = Depends(verify_token), db: Session = Depends(get_db)):
    """Revoke share link"""
    share = db.query(models.Share).filter_by(id=share_id).first()
    if not share:
        raise HTTPException(status_code=404, detail="Share not found")
    
    # Ownership check
    if share.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    share.status = "revoked"
    db.commit()
    
    return {"message": "Share link revoked"}

@app.get("/s/{token}")
def access_share_redirect(token: str):
    """
    Public share access - Redirects to frontend with share_token parameter.
    The frontend will then fetch data from the public API endpoint.
    """
    # Redirect to frontend with share_token
    # Frontend should be configured to handle this URL parameter
    return RedirectResponse(url=f"/?share_token={token}")


@app.get(f"{settings.API_V1_STR}/shares/public/{{token}}")
def get_public_share_data(token: str, db: Session = Depends(get_db)):
    """
    Public API endpoint to fetch share data (no auth required).
    Used by frontend after redirect from /s/{token}.
    """
    share = db.query(models.Share).filter_by(token=token).first()
    
    if not share:
        raise HTTPException(status_code=404, detail="Share not found")
    
    if share.status == "revoked":
        raise HTTPException(status_code=403, detail="Share link has been revoked")
    
    if share.expires_at < datetime.now(timezone.utc):
        # Update status to expired
        share.status = "expired"
        db.commit()
        raise HTTPException(status_code=403, detail="Share link has expired")
    
    if share.status != "active":
        raise HTTPException(status_code=403, detail=f"Share link is {share.status}")
    
    # Increment views
    share.views += 1
    db.commit()
    
    # Get records with full details
    records = db.query(models.Record)\
        .filter(models.Record.id.in_(share.record_ids))\
        .all()
    
    return {
        "share_name": share.name,
        "recipient": share.recipient_name,
        "records": [schemas.Record.from_orm(r) for r in records],
        "expires_at": share.expires_at,
        "views": share.views
    }

# ============================================================================
# TIMELINE ROUTES
# ============================================================================

@app.get(f"{settings.API_V1_STR}/timeline", response_model=List[schemas.TimelineEventsByDate])
def get_timeline(
    profile_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get timeline events for profile grouped by date"""
    events = db.query(models.TimelineEvent)\
        .filter(models.TimelineEvent.profile_id == profile_id)\
        .order_by(models.TimelineEvent.created_at.desc())\
        .limit(limit)\
        .all()
    
    # Group by month
    grouped = {}
    for event in events:
        month_key = event.created_at.strftime("%B %Y")
        if month_key not in grouped:
            grouped[month_key] = []
        grouped[month_key].append(event)
    
    return [
        {"date": date, "events": events}
        for date, events in grouped.items()
    ]

# ============================================================================
# USER NOTIFICATIONS ROUTES
# ============================================================================

def format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time string"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    diff = now - dt
    
    if diff.total_seconds() < 60:
        return "Just now"
    elif diff.total_seconds() < 3600:
        mins = int(diff.total_seconds() / 60)
        return f"{mins} min{'s' if mins > 1 else ''} ago"
    elif diff.total_seconds() < 86400:
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = diff.days
        return f"{days} day{'s' if days > 1 else ''} ago"


@app.get(f"{settings.API_V1_STR}/notifications", response_model=List[schemas.NotificationItem])
def get_user_notifications(
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db),
    limit: int = 20
):
    """Get notifications for the current user (for bell icon)"""
    notifications = db.query(models.Notification)\
        .filter(models.Notification.user_id == user.user_id)\
        .order_by(models.Notification.created_at.desc())\
        .limit(limit)\
        .all()
    
    return [
        schemas.NotificationItem(
            id=n.id,
            title=n.title,
            message=n.message,
            type=n.type,
            read=n.read,
            time=format_relative_time(n.created_at)
        )
        for n in notifications
    ]


@app.get(f"{settings.API_V1_STR}/notifications/unread-count")
def get_unread_count(
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get count of unread notifications for badge"""
    from sqlalchemy import func as sqlfunc
    count = db.query(sqlfunc.count(models.Notification.id))\
        .filter(models.Notification.user_id == user.user_id)\
        .filter(models.Notification.read == False)\
        .scalar() or 0
    
    return {"unread_count": count}


@app.patch(f"{settings.API_V1_STR}/notifications/{{notification_id}}/read")
def mark_notification_read(
    notification_id: str,
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Mark a notification as read"""
    notification = db.query(models.Notification)\
        .filter(models.Notification.id == notification_id)\
        .filter(models.Notification.user_id == user.user_id)\
        .first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    notification.read = True
    db.commit()
    
    return {"message": "Notification marked as read"}


@app.patch(f"{settings.API_V1_STR}/notifications/read-all")
def mark_all_notifications_read(
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Mark all notifications as read for the current user"""
    db.query(models.Notification)\
        .filter(models.Notification.user_id == user.user_id)\
        .filter(models.Notification.read == False)\
        .update({"read": True})
    db.commit()
    
    return {"message": "All notifications marked as read"}


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
