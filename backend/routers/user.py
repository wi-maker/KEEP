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

@router.get("/notifications", response_model=List[schemas.NotificationItem])
async def get_user_notifications(
    user: TokenPayload = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Get combined list of system announcements and personal notifications.
    Used for the notification bell.
    """
    # 1. Fetch User Notifications
    notifs = db.query(models.Notification)\
        .filter(models.Notification.user_id == user.user_id)\
        .order_by(models.Notification.created_at.desc())\
        .limit(20)\
        .all()
        
    # 2. Fetch Active Announcements
    anns = db.query(models.Announcement)\
        .filter(models.Announcement.is_active == True)\
        .order_by(models.Announcement.created_at.desc())\
        .limit(5)\
        .all()
        
    combined = []
    
    # Add Announcements (showing as unread generally, or frontend handles logic)
    for a in anns:
        combined.append(schemas.NotificationItem(
            id=str(a.id),
            title=a.title,
            message=a.message,
            type="announcement",
            read=False, # Announcements are effectively unread until dismissed on frontend
            time=format_relative_time(a.created_at)
        ))
        
    # Add User Notifications
    for n in notifs:
        combined.append(schemas.NotificationItem(
            id=str(n.id),
            title=n.title,
            message=n.message,
            type=n.type,
            read=n.read,
            time=format_relative_time(n.created_at)
        ))
        
    # Re-sort by time (approximation as time string loses precision, but list is short)
    # Actually just returning combined list is fine, frontend can sort or just display announcements first
    
    return combined
