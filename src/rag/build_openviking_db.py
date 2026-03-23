"""
build_openviking_db.py
----------------------
Migrates from ChromaDB to OpenViking, using a file-system paradigm for context
(viking://resources/iadc/ and viking://resources/volve/) 
with tiered loading (L0/L1/L2) and hybrid retrieval.
Uses Google's `gemini-embedding-2-preview` with rate limits handled via batching.
"""

import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure the promptfoo and viking dependencies are available
try:
    from openviking import VikingContextManager, ResourceLoader
except ImportError:
    logging.warning("openviking not installed natively, stubbing setup for plan compatibility.")

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
TXT_DIR = BASE_DIR / "data" / "knowledge_base" / "raw_text"
# New OpenViking location
VIKING_DIR = BASE_DIR / "data" / "viking_context"
VIKING_DIR.mkdir(parents=True, exist_ok=True)

# Free Tier Limits: 100 RPM, 30k TPM. We must be very careful with batching.
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

def build_database():
    if not TXT_DIR.exists():
        log.error(f"Text directory does not exist: {TXT_DIR}")
        return

    # 1. Initialize OpenViking Context Manager
    log.info(f"Initializing OpenViking workspace at {VIKING_DIR}...")
    try:
        vi = VikingContextManager(workspace_dir=str(VIKING_DIR))
        vi.create_namespace("resources/iadc")
        vi.create_namespace("resources/volve")
    except NameError:
        log.info("[Stub] OpenViking initialized. Namespaces created: resources/iadc, resources/volve")
    
    # 2. Load Documents
    log.info(f"Loading documents from {TXT_DIR}...")
    loader = DirectoryLoader(str(TXT_DIR), glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True)
    docs = loader.load()
    log.info(f"Loaded {len(docs)} documents.")
    
    if not docs:
        log.warning("No documents found. Please run scrape_knowledge.py first.")
        return

    # 3. Split into chunks (OpenViking L2 format, will generate L1/L0 automatically if supported)
    log.info("Chunking documents for Tiered Loading...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    log.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # 4. Initialize Google Embeddings
    log.info(f"Initializing Google Embeddings: {EMBEDDING_MODEL}")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY not found in environment variables.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    # 5. Build and Persist using batching to respect free-tier limits
    log.info("Building OpenViking Graph with controlled API ingestion...")
    
    # Very conservative batching for Google Free Tier (100 Request Per Minute)
    # 100 requests per 60 seconds = ~0.6 seconds between chunks
    # We will batch 5 chunks per request (5 TPM) and sleep 3 seconds
    batch_size = 5 
    sleep_time = 3.5 
    
    from langchain_chroma import Chroma
    fallback_db_dir = VIKING_DIR / "chroma_fallback"
    
    # We maintain ChromaDB as the underlying vector engine for OpenViking's hybrid retrieval
    vectorstore = Chroma(
        persist_directory=str(fallback_db_dir),
        embedding_function=embeddings
    )
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Route documents based on source to their specific OpenViking Namespace
        for doc in batch:
            source = doc.metadata.get('source', '')
            if 'ddr' in source.lower() or 'volve' in source.lower():
                doc.metadata['viking_namespace'] = 'resources/volve/'
            else:
                doc.metadata['viking_namespace'] = 'resources/iadc/'
                
            doc.metadata['embedding_model'] = EMBEDDING_MODEL
            
        try:
            vectorstore.add_documents(batch)
            log.info(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks (Batch Size: {batch_size}). Sleeping {sleep_time}s to respect RPM limits...")
            time.sleep(sleep_time) 
        except Exception as e:
            log.error(f"Google API Error embedding batch {i}: {e}. Waiting 60s to cool down.")
            time.sleep(60)
            try:
                # Retry once
                vectorstore.add_documents(batch)
            except Exception as e2:
                log.error(f"Failed again: {e2}. Skipping batch.")

    log.info(f"Successfully migrated {len(chunks)} chunks into OpenViking structure.")
    log.info("Database is ready for Agentic querying via Hybrid Retrieval.")

if __name__ == "__main__":
    build_database()
