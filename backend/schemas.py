from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

# --- User Schemas ---
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: Optional[str] = None

class User(UserBase):
    id: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# --- Profile Schemas ---
class ProfileBase(BaseModel):
    name: str
    relation: str  # Me, Spouse, Child, Parent

class ProfileCreate(ProfileBase):
    avatar_seed: Optional[str] = None

class Profile(ProfileBase):
    id: str
    user_id: str
    avatar_seed: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# --- Record Schemas ---
class RecordFileSchema(BaseModel):
    id: str
    file_path: str
    file_type: str
    original_filename: str
    file_size: Optional[int]

    class Config:
        from_attributes = True

class RecordAnalysisSchema(BaseModel):
    id: str
    summary: Optional[str]
    key_findings: Optional[List[str]]
    confidence_score: Optional[float]
    doctor_questions: Optional[List[str]]
    medical_sources_used: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True

class RecordBase(BaseModel):
    title: str
    record_date: Optional[datetime] = None
    doctor_name: Optional[str] = None
    facility_name: Optional[str] = None

class RecordCreate(RecordBase):
    profile_id: str

class RecordUpdate(BaseModel):
    title: Optional[str] = None
    record_date: Optional[datetime] = None
    doctor_name: Optional[str] = None
    facility_name: Optional[str] = None

class Record(RecordBase):
    id: str
    profile_id: str
    record_type: Optional[str]
    status: str
    created_at: datetime
    updated_at: Optional[datetime]
    files: List[RecordFileSchema] = []
    analysis: Optional[RecordAnalysisSchema] = None

    class Config:
        from_attributes = True

class RecordList(BaseModel):
    id: str
    title: str
    record_date: Optional[datetime]
    record_type: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Upload Schema ---
class UploadResponse(BaseModel):
    record_id: str
    status: str
    message: str

# --- Chat Schemas ---
class ChatRequest(BaseModel):
    profile_id: str
    message: str

class ChatSource(BaseModel):
    type: str  # "user_document" or "medical_knowledge"
    title: str  # Record title or source name
    excerpt: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    sources: List[ChatSource] = []
    confidence: Optional[float] = None

class ChatHistoryItem(BaseModel):
    id: str
    role: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True

# --- Share Schemas ---
class ShareCreate(BaseModel):
    name: str
    recipient_name: str
    record_ids: List[str]
    expiry_hours: int = 24  # Default 24 hours

class ShareResponse(BaseModel):
    id: str
    name: str
    recipient_name: str
    record_ids: List[str]
    token: str
    share_url: str
    expires_at: datetime
    status: str
    views: int
    created_at: datetime

    class Config:
        from_attributes = True

class ShareList(BaseModel):
    id: str
    name: str
    recipient_name: str
    record_count: int
    status: str
    expires_at: datetime
    views: int

# --- Timeline Schemas ---
class TimelineEvent(BaseModel):
    id: str
    event_type: str
    event_title: str
    related_record_id: Optional[str]
    event_metadata: Optional[dict]
    created_at: datetime

    class Config:
        from_attributes = True

class TimelineEventsByDate(BaseModel):
    date: str  # e.g., "October 2023"
    events: List[TimelineEvent]

# --- Auth Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
