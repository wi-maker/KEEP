from sqlalchemy import Column, String, Boolean, ForeignKey, DateTime, Text, JSON, Float, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    profiles = relationship("Profile", back_populates="owner", cascade="all, delete-orphan")
    shares = relationship("Share", back_populates="creator", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")

class Profile(Base):
    __tablename__ = "profiles"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    relation = Column(String, nullable=False)  # Me, Spouse, Child, Parent
    avatar_seed = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="profiles")
    records = relationship("Record", back_populates="profile", cascade="all, delete-orphan")
    timeline_events = relationship("TimelineEvent", back_populates="profile", cascade="all, delete-orphan")

class Record(Base):
    __tablename__ = "records"

    id = Column(String, primary_key=True, default=generate_uuid)
    profile_id = Column(String, ForeignKey("profiles.id", ondelete="CASCADE"))
    
    title = Column(String, nullable=False)
    record_date = Column(DateTime)
    record_type = Column(String)  # Lab Report, Vaccination, Imaging, etc.
    doctor_name = Column(String)
    facility_name = Column(String)
    
    status = Column(String, default="Pending")  # Pending, Processing, Analyzed, Error
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    profile = relationship("Profile", back_populates="records")
    files = relationship("RecordFile", back_populates="record", cascade="all, delete-orphan")
    texts = relationship("RecordText", back_populates="record", cascade="all, delete-orphan")
    analysis = relationship("RecordAnalysis", uselist=False, back_populates="record", cascade="all, delete-orphan")

class RecordFile(Base):
    __tablename__ = "record_files"

    id = Column(String, primary_key=True, default=generate_uuid)
    record_id = Column(String, ForeignKey("records.id", ondelete="CASCADE"))
    file_path = Column(String, nullable=False)
    file_type = Column(String)  # pdf, jpg, png
    original_filename = Column(String)
    file_size = Column(Integer)
    
    record = relationship("Record", back_populates="files")

class RecordText(Base):
    __tablename__ = "record_text"

    id = Column(String, primary_key=True, default=generate_uuid)
    record_id = Column(String, ForeignKey("records.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    page_number = Column(Integer)
    
    record = relationship("Record", back_populates="texts")

class RecordAnalysis(Base):
    __tablename__ = "record_analysis"

    id = Column(String, primary_key=True, default=generate_uuid)
    record_id = Column(String, ForeignKey("records.id", ondelete="CASCADE"), unique=True)
    
    summary = Column(Text)  # Plain language summary
    key_findings = Column(JSON)  # List of strings
    confidence_score = Column(Float)  # 0-100
    doctor_questions = Column(JSON)  # List of strings
    
    medical_sources_used = Column(JSON)  # List of source names/URLs
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    record = relationship("Record", back_populates="analysis")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    profile_id = Column(String, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=True)
    role = Column(String)  # user, assistant
    content = Column(Text)
    context_used = Column(JSON, nullable=True)  # What was retrieved for context
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="chat_history")

class Share(Base):
    __tablename__ = "shares"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    
    name = Column(String)  # Packet name
    recipient_name = Column(String)
    record_ids = Column(JSON)  # List of record IDs
    
    token = Column(String, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True))
    status = Column(String, default="active")  # active, expired, revoked
    views = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    creator = relationship("User", back_populates="shares")

class TimelineEvent(Base):
    __tablename__ = "timeline_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    profile_id = Column(String, ForeignKey("profiles.id", ondelete="CASCADE"))
    
    event_type = Column(String)  # upload, analysis, share, profile_update
    event_title = Column(String)
    related_record_id = Column(String, nullable=True)
    event_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved in SQLAlchemy)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    profile = relationship("Profile", back_populates="timeline_events")
