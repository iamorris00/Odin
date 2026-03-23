"""
test_retrieval.py
-----------------
Tests the locally built ChromaDB vector store
using the sentence-transformer embeddings.
"""

import sys
from pathlib import Path
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DB_DIR = BASE_DIR / "data" / "knowledge_base" / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

def test_query(query: str, k: int = 3):
    if not DB_DIR.exists():
        log.error("ChromaDB not found. Run build_vector_db.py first.")
        return
        
    log.info(f"Loading BGE model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    log.info(f"Loading Chroma database from {DB_DIR}...")
    vectorstore = Chroma(
        persist_directory=str(DB_DIR), 
        embedding_function=embeddings
    )
    
    log.info(f"\n--- QUERY: '{query}' ---")
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    if not results:
        log.warning("No results found.")
        return
        
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        log.info(f"\n[Result {i} | SimScore: {score:.4f} | Source: {Path(source).name}]")
        # Print a snippet of the page content
        content = doc.page_content.replace('\n', ' ')
        log.info(f"{content[:500]}..." if len(content) > 500 else content)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What causes stuck pipe during a drilling operation?"
        log.info("No query provided. Using default:")
        
    test_query(query)
