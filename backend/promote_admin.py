"""
Script to promote a user to admin
Usage: python promote_admin.py <email>
"""
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from models import User

def promote_user(email: str):
    print(f"Connecting to database: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'SQLite/Local'}")
    
    # Fix for Railway/Supabase URLs if needed
    db_url = settings.DATABASE_URL
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"Error: User with email '{email}' not found.")
            return
        
        if user.is_admin:
            print(f"User '{email}' is already an admin.")
            return
            
        user.is_admin = True
        db.commit()
        print(f"Success! User '{email}' has been promoted to admin.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python promote_admin.py <email>")
        sys.exit(1)
        
    email = sys.argv[1]
    promote_user(email)
