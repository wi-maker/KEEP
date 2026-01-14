import os
from supabase import create_client, Client
from fastapi import UploadFile

# Initialize Supabase (Use service key for backend operations if possible, or anon key)
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

BUCKET_NAME = "records"

async def upload_file_to_supabase(file: UploadFile, file_path: str) -> str:
    """
    Uploads a file to Supabase Storage and returns the public URL.
    file_path: e.g., "user_123/record_abc.pdf"
    """
    try:
        file_content = await file.read()
        # Upload to Supabase
        supabase.storage.from_(BUCKET_NAME).upload(
            file_path,
            file_content,
            {"content-type": file.content_type}
        )
        
        # Get Public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        return public_url
    except Exception as e:
        print(f"Upload Error: {e}")
        # If it fails (e.g. file exists), try to update or return None
        return None