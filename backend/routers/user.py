from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone

import models
import schemas
from db import get_db
from auth import verify_token, TokenPayload

router = APIRouter(prefix="/user", tags=["user"])

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

@router.get("/notifications")
async def get_user_notifications(
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Get combined list of system announcements and personal notifications.
    Used for the notification bell.
    """
    # 1. Fetch Active Global Announcements
    # We remove strict is_active filter just in case, or ensure defaults are respected
    announcements = db.query(models.Announcement)\
        .order_by(models.Announcement.created_at.desc())\
        .limit(5)\
        .all()
        
    # 2. Fetch Personal Notifications
    personal = db.query(models.Notification)\
        .filter(models.Notification.user_id == user.user_id)\
        .order_by(models.Notification.created_at.desc())\
        .limit(20)\
        .all()
        
    results = []
    
    # Add Announcements (Global)
    for a in announcements:
        # Check if active if column exists/is populated, otherwise default to show
        if hasattr(a, 'is_active') and a.is_active is False:
            continue
            
        results.append({
            "id": str(a.id),
            "title": a.title,
            "message": a.message,
            "type": "announcement",
            "time": format_relative_time(a.created_at),
            "read": False # Announcements always unread/highlighted
        })
        
    # Add Personal Notifications
    for n in personal:
        results.append({
            "id": str(n.id),
            "title": n.title,
            "message": n.message,
            "type": n.type or "info",
            "time": format_relative_time(n.created_at),
            "read": n.read
        })
    
    return results

