import os
import io
import re
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from supabase import create_client
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from pypdf import PdfReader

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def download_file(service, file_id, mime_type):
    if 'google-apps' in mime_type:
        request = service.files().export_media(
            fileId=file_id,
            mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    else:
        request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def extract_text_from_bytes(fh, mime_type, filename):
    if filename.endswith('.pdf'):
        reader = PdfReader(fh)
        return "\n".join([page.extract_text() for page in reader.pages])
    else:
        doc = Document(fh)
        return "\n".join([p.text for p in doc.paragraphs])

def check_upload_status(supabase, file_id, filename, modified_time):
    """'new' | 'unchanged' | 'modified' 반환"""
    result = supabase.table("documents").select("metadata").eq("metadata->>file_id", file_id).limit(1).execute()
    if not result.data:
        # 기존 데이터 호환: 파일명으로 fallback
        result = supabase.table("documents").select("metadata").eq("metadata->>source", filename).limit(1).execute()
    if not result.data:
        return "new"
    stored_time = result.data[0]["metadata"].get("modified_time")
    return "unchanged" if stored_time == modified_time else "modified"

def delete_from_supabase(supabase, file_id, filename):
    supabase.table("documents").delete().eq("metadata->>file_id", file_id).execute()
    # 기존 데이터 호환: 파일명으로도 삭제
    supabase.table("documents").delete().eq("metadata->>source", filename).execute()

def extract_speakers(text):
    return sorted(set(re.findall(r'^([^\s:]{1,10}):', text, re.MULTILINE)))

def upload_to_supabase(text, filename, file_id=None, modified_time=None):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    status = check_upload_status(supabase, file_id, filename, modified_time)
    if status == "unchanged":
        print(f"⏭️ {filename} → 변경 없음, 건너뜀")
        return
    if status == "modified":
        print(f"🔄 {filename} → 수정 감지, 기존 데이터 삭제 후 재업로드")
        delete_from_supabase(supabase, file_id, filename)
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    file_speakers = extract_speakers(text)
    base_metadata = {"source": filename, "file_id": file_id, "modified_time": modified_time}
    for chunk in chunks:
        chunk_speakers = extract_speakers(chunk)
        metadata = {**base_metadata, "speakers": chunk_speakers or file_speakers}
        embedding = model.encode(chunk).tolist()
        supabase.table("documents").insert({
            "content": chunk,
            "metadata": metadata,
            "embedding": embedding
        }).execute()
    print(f"✅ {filename} → {len(chunks)}개 조각 저장됨 (화자: {file_speakers})")

def sync_drive_folder(folder_id):
    service = get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType, modifiedTime)"
    ).execute()
    files = results.get('files', [])
    print(f"📁 {len(files)}개 파일 발견")
    for file in files:
        name = file['name']
        if any(name.endswith(ext) for ext in ['.docx', '.pdf', '.txt']) or 'google-apps.document' in file['mimeType']:
            print(f"📄 처리 중: {name}")
            fh = download_file(service, file['id'], file['mimeType'])
            text = extract_text_from_bytes(fh, file['mimeType'], name)
            if text.strip():
                upload_to_supabase(text, name, file['id'], file.get('modifiedTime'))

# 구글 드라이브 폴더 ID 입력
FOLDER_ID = "1-o0LV8a4sOeP9ruvKmxsKDm1DuaIPBmv"
sync_drive_folder(FOLDER_ID)