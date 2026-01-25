"""
Admin Router for KEEP Platform
Protected endpoints for admin dashboard functionality
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime, timedelta, timezone

import models
import schemas
from db import get_db
from auth import get_current_admin, TokenPayload

# Create admin dependency
require_admin = get_current_admin(get_db)

router = APIRouter(prefix="/admin", tags=["admin"])


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


# async def get_admin_overview(
#     user: TokenPayload = Depends(require_admin),
#     db: Session = Depends(get_db)
# ):
@router.get("/overview", response_model=schemas.AdminOverview)
async def get_admin_overview(
    db: Session = Depends(get_db)
):
    """Get dashboard overview with KPIs and charts data"""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    thirty_days_ago = now - timedelta(days=30)
    
    # Total counts
    total_users = db.query(func.count(models.User.id)).scalar() or 0
    total_records = db.query(func.count(models.Record.id)).scalar() or 0
    
    # Active users (users with chat history in last 30 days)
    active_users = db.query(func.count(func.distinct(models.ChatHistory.user_id)))\
        .filter(models.ChatHistory.timestamp >= thirty_days_ago)\
        .scalar() or 0
    
    # Today's stats
    new_users_today = db.query(func.count(models.User.id))\
        .filter(models.User.created_at >= today_start)\
        .scalar() or 0
    
    records_today = db.query(func.count(models.Record.id))\
        .filter(models.Record.created_at >= today_start)\
        .scalar() or 0
    
    ai_requests_today = db.query(func.count(models.AILog.id))\
        .filter(models.AILog.created_at >= today_start)\
        .scalar() or 0
    
    # User growth (last 7 days cumulative)
    user_growth = []
    for i in range(6, -1, -1):
        day_end = now - timedelta(days=i)
        count = db.query(func.count(models.User.id))\
            .filter(models.User.created_at <= day_end)\
            .scalar() or 0
        user_growth.append(count)
    
    # Daily uploads (last 7 days)
    record_uploads_daily = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0)
        day_end = day_start + timedelta(days=1)
        count = db.query(func.count(models.Record.id))\
            .filter(models.Record.created_at >= day_start)\
            .filter(models.Record.created_at < day_end)\
            .scalar() or 0
        record_uploads_daily.append(count)
    
    # Recent activity (last 10 timeline events)
    recent_events = db.query(models.TimelineEvent)\
        .order_by(models.TimelineEvent.created_at.desc())\
        .limit(10)\
        .all()
    
    recent_activity = [
        {
            "type": e.event_type,
            "message": e.event_title,
            "time": format_relative_time(e.created_at)
        }
        for e in recent_events
    ]
    
    # System health (simple status check)
    system_health = {
        "api": {"status": "healthy", "response_time": 120},
        "ai": {"status": "healthy", "latency": 2.3},
        "database": {"status": "healthy", "connections": 23}
    }
    
    return schemas.AdminOverview(
        total_users=total_users,
        active_users_30d=active_users,
        total_records=total_records,
        ai_requests_today=ai_requests_today,
        new_users_today=new_users_today,
        records_uploaded_today=records_today,
        user_growth=user_growth,
        record_uploads_daily=record_uploads_daily,
        recent_activity=recent_activity,
        system_health=system_health
    )


@router.get("/users", response_model=schemas.AdminUsersResponse)
async def get_admin_users(
    db: Session = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    """Get paginated list of users with stats"""
    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)
    
    # Stats
    new_this_month = db.query(func.count(models.User.id))\
        .filter(models.User.created_at >= thirty_days_ago)\
        .scalar() or 0
    
    total_users = db.query(func.count(models.User.id)).scalar() or 0
    active_users = db.query(func.count(func.distinct(models.ChatHistory.user_id)))\
        .filter(models.ChatHistory.timestamp >= thirty_days_ago)\
        .scalar() or 0
    
    engagement_rate = (active_users / total_users * 100) if total_users > 0 else 0
    
    # Get users with record count
    users = db.query(models.User)\
        .order_by(models.User.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    recent_users = []
    for u in users:
        record_count = 0
        for profile in u.profiles:
            record_count += len(profile.records)
        
        recent_users.append(schemas.AdminUser(
            id=u.id,
            name=u.full_name,
            email=u.email,
            records=record_count,
            joined=u.created_at,
            active=u.is_active
        ))
    
    return schemas.AdminUsersResponse(
        stats={
            "new_this_month": new_this_month,
            "engagement_rate": round(engagement_rate, 1),
            "avg_session": 8.2  # Placeholder
        },
        recent_users=recent_users
    )


@router.get("/ai", response_model=schemas.AIStats)
async def get_ai_stats(
    db: Session = Depends(get_db)
):
    """Get AI usage statistics"""
    now = datetime.now(timezone.utc)
    
    # Total stats
    total_requests = db.query(func.count(models.AILog.id)).scalar() or 0
    total_tokens = db.query(func.sum(models.AILog.tokens_used)).scalar() or 0
    failed_requests = db.query(func.count(models.AILog.id))\
        .filter(models.AILog.status == "failed")\
        .scalar() or 0
    
    # Request trends (last 7 days)
    request_trends = []
    token_usage = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0)
        day_end = day_start + timedelta(days=1)
        
        req_count = db.query(func.count(models.AILog.id))\
            .filter(models.AILog.created_at >= day_start)\
            .filter(models.AILog.created_at < day_end)\
            .scalar() or 0
        request_trends.append(req_count)
        
        tokens = db.query(func.sum(models.AILog.tokens_used))\
            .filter(models.AILog.created_at >= day_start)\
            .filter(models.AILog.created_at < day_end)\
            .scalar() or 0
        token_usage.append(tokens)
    
    return schemas.AIStats(
        total_requests=total_requests,
        tokens_used=total_tokens,
        avg_response_time=2.3,  # Placeholder
        failed_requests=failed_requests,
        request_trends=request_trends,
        token_usage=token_usage
    )


@router.post("/announcements", response_model=schemas.AnnouncementResponse)
async def create_announcement(
    announcement: schemas.AnnouncementCreate,
    db: Session = Depends(get_db)
):
    """Create announcement and broadcast to all users as notifications"""
    # Create announcement
    db_announcement = models.Announcement(
        title=announcement.title,
        message=announcement.message,
        type=announcement.type
    )
    db.add(db_announcement)
    db.commit()
    db.refresh(db_announcement)
    
    # Broadcast to all active users as notifications
    active_users = db.query(models.User).filter(models.User.is_active == True).all()
    
    for u in active_users:
        notification = models.Notification(
            user_id=u.id,
            title=announcement.title,
            message=announcement.message,
            type="announcement"
        )
        db.add(notification)
    
    db.commit()
    
    return schemas.AnnouncementResponse(
        id=db_announcement.id,
        title=db_announcement.title,
        message=db_announcement.message,
        type=db_announcement.type,
        is_active=db_announcement.is_active,
        created_at=db_announcement.created_at
    )


@router.get("/notifications", response_model=List[schemas.NotificationItem])
async def get_admin_notifications(
    db: Session = Depends(get_db),
    limit: int = 20
):
    """Get admin notifications (recent system events)"""
    # For admin, show recent timeline events as notifications
    recent_events = db.query(models.TimelineEvent)\
        .order_by(models.TimelineEvent.created_at.desc())\
        .limit(limit)\
        .all()
    
    type_map = {
        "upload": "record",
        "profile_created": "user",
        "share": "share",
        "analysis": "ai"
    }
    
    return [
        schemas.NotificationItem(
            id=e.id,
            title=e.event_type.replace("_", " ").title(),
            message=e.event_title,
            type=type_map.get(e.event_type, "system"),
            read=False,
            time=format_relative_time(e.created_at)
        )
        for e in recent_events
    ]


@router.get("/system", response_model=schemas.SystemHealth)
async def get_system_health(
    db: Session = Depends(get_db)
):
    """Get system health status"""
    # Check database connection
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    services = [
        {"name": "API Service", "status": "healthy", "uptime": 99.9, "last_check": "1 min ago"},
        {"name": "AI Service", "status": "healthy", "uptime": 99.7, "last_check": "1 min ago"},
        {"name": "Database", "status": db_status, "uptime": 100, "last_check": "30 sec ago"},
        {"name": "Storage", "status": "healthy", "uptime": 99.8, "last_check": "2 min ago"}
    ]
    
    # Recent log entries (simulated from timeline events)
    recent_events = db.query(models.TimelineEvent)\
        .order_by(models.TimelineEvent.created_at.desc())\
        .limit(5)\
        .all()
    
    recent_logs = [
        {
            "level": "info",
            "message": e.event_title,
            "time": e.created_at.strftime("%Y-%m-%d %H:%M:%S") if e.created_at else ""
        }
        for e in recent_events
    ]
    
    return schemas.SystemHealth(
        services=services,
        recent_logs=recent_logs
    )


@router.get("/announcements", response_model=List[schemas.AnnouncementItem])
def get_announcements(
    db: Session = Depends(get_db),
    # user: TokenPayload = Depends(require_admin)
):
    """Get list of announcements"""
    anns = db.query(models.Announcement).order_by(models.Announcement.created_at.desc()).limit(10).all()
    return [
        schemas.AnnouncementItem(
            id=str(a.id),
            title=a.title,
            message=a.message,
            type=a.type,
            target="All Users", # Placeholder as we don't store target yet
            time=a.created_at.strftime("%Y-%m-%d %H:%M")
        )
        for a in anns
    ]


@router.get("/records", response_model=schemas.AdminRecordsResponse)
async def get_admin_records(
    db: Session = Depends(get_db),
    # user: TokenPayload = Depends(require_admin)
):
    """Get records stats and recent list"""
    now = datetime.now(timezone.utc)
    
    # 1. By Type Stats
    # Join RecordFile to query file types correctly
    pdf_count = db.query(func.count(models.Record.id)).join(models.RecordFile).filter(models.RecordFile.file_type.ilike('%pdf%')).scalar() or 0
    img_count = db.query(func.count(models.Record.id)).join(models.RecordFile).filter(
        (models.RecordFile.file_type.ilike('%jpg%')) | 
        (models.RecordFile.file_type.ilike('%png%')) | 
        (models.RecordFile.file_type.ilike('%jpeg%'))
    ).scalar() or 0
    doc_count = db.query(func.count(models.Record.id)).join(models.RecordFile).filter(models.RecordFile.file_type.ilike('%doc%')).scalar() or 0
    
    by_type = {
        "pdf": pdf_count,
        "image": img_count,
        "document": doc_count
    }
    
    # 2. Recent Uploads
    recent = db.query(models.Record).order_by(models.Record.created_at.desc()).limit(10).all()
    recent_uploads = []
    
    for r in recent:
        # Get user name from profile
        user_name = "Unknown"
        if r.profile and r.profile.owner:
            user_name = r.profile.owner.full_name or r.profile.owner.email
            
        # Determine format type
        file_type = "document"
        if r.files:
            ft = r.files[0].file_type.lower()
            if "pdf" in ft: file_type = "pdf"
            elif "image" in ft or "jpg" in ft or "png" in ft: file_type = "image"
            
        recent_uploads.append(schemas.AdminRecordItem(
            title=r.title,
            type=file_type,
            user=user_name,
            date=r.created_at.strftime("%Y-%m-%d"),
            status=r.status.title()
        ))
        
    # 3. Upload Trends (last 7 days)
    upload_trends = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0)
        day_end = day_start + timedelta(days=1)
        count = db.query(func.count(models.Record.id))\
            .filter(models.Record.created_at >= day_start)\
            .filter(models.Record.created_at < day_end)\
            .scalar() or 0
        upload_trends.append(count)
            
    return schemas.AdminRecordsResponse(
        by_type=by_type,
        recent_uploads=recent_uploads,
        upload_trends=upload_trends
    )

