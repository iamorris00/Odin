"""
build_vector_db.py
------------------
Reads raw scraped text files, chunks them, and embeds them into ChromaDB 
using a local open-source model (all-MiniLM-L6-v2) to avoid API limits.
"""

import os
from pathlib import Path
import logging
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
TXT_DIR = BASE_DIR / "data" / "knowledge_base" / "raw_text"
DB_DIR = BASE_DIR / "data" / "knowledge_base" / "chroma_db"
EMBEDDING_MODEL = "Octen/Octen-Embedding-0.6B"

def build_database():
    if not TXT_DIR.exists():
        log.error(f"Text directory does not exist: {TXT_DIR}")
        return

    # Clear old dimension index if we are changing models
    if DB_DIR.exists():
        log.info(f"Clearing existing database at {DB_DIR} to avoid dimension mismatch...")
        import shutil
        shutil.rmtree(DB_DIR)
        
    # 1. Load Documents
    log.info(f"Loading documents from {TXT_DIR}...")
    loader = DirectoryLoader(str(TXT_DIR), glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True)
    docs = loader.load()
    log.info(f"Loaded {len(docs)} documents.")
    
    if not docs:
        log.warning("No documents found. Please run scrape_knowledge.py first.")
        return

    # 2. Split into chunks
    log.info("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    log.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # 3. Initialize HuggingFaceEmbeddings using GPU VRAM
    log.info(f"Initializing powerful model: {EMBEDDING_MODEL}")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Build and Persist ChromaDB
    log.info(f"Building and persisting ChromaDB at {DB_DIR}...")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize an empty vector store
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )
    
    batch_size = 200 # Process 200 chunks at a time for safety
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        log.info(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
    
    log.info(f"Successfully embedded {len(chunks)} chunks into ChromaDB.")
    log.info("Database is ready for Agentic querying.")

if __name__ == "__main__":
    build_database()
