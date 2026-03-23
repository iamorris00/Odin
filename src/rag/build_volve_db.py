"""
build_volve_db.py
-----------------
Builds a combined Volve History & Geophysics Vector DB.
Includes:
1. Structured DDR Activity Narratives
2. Geological Formation Picks (Geophysical Interpretations)
"""

import os
import time
import shutil
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
DB_DIR = BASE_DIR / "data" / "knowledge_base" / "volve_ddr_history"
DDR_CSV = DATA_DIR / "ddr" / "_ddr_all_activities.csv"
PICKS_CSV = DATA_DIR / "serialized_text" / "well_picks_narratives.csv"

def build_combined_db():
    documents = []

    # 1. Ingest DDR Activities
    if DDR_CSV.exists():
        logger.info(f"Loading DDR activities from {DDR_CSV}...")
        df_ddr = pd.read_csv(DDR_CSV).fillna("")
        for idx, row in tqdm(df_ddr.iterrows(), total=len(df_ddr), desc="DDR"):
            well = str(row.get("well_name", ""))
            date = str(row.get("act_start", ""))[:10]
            comm = str(row.get("comments", "")).strip()
            state = str(row.get("state", ""))
            if not comm and state == "ok": continue
            
            content = f"Date: {date}\nWell: {well}\nActivity: {row.get('activity_code','')}\nDepth: {row.get('md_m','')}m\nComments: {comm}"
            metadata = {"source": "DDR", "well": well, "date": date, "type": "activity"}
            documents.append(Document(page_content=content, metadata=metadata))
    
    # 2. Ingest Well Picks (Geophysics)
    if PICKS_CSV.exists():
        logger.info(f"Loading Well Picks from {PICKS_CSV}...")
        df_picks = pd.read_csv(PICKS_CSV)
        for idx, row in tqdm(df_picks.iterrows(), total=len(df_picks), desc="Picks"):
            content = row["text"]
            # Extract well name from narrative for metadata if possible
            well_match = re.search(r"Well ([\w\s/-]+),", content)
            well = well_match.group(1) if well_match else "Unknown"
            metadata = {"source": "Geophysics", "well": well, "type": "formation_pick"}
            documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        logger.error("No documents found to index.")
        return

    # Clear existing
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    # Embeddings
    logger.info("Initializing HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Octen/Octen-Embedding-0.6B",
        model_kwargs={'device': 'cuda', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Vector Store
    logger.info(f"Building combined Vector DB at {DB_DIR} with {len(documents)} docs...")
    vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    
    batch_size = 1000
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing"):
        vectorstore.add_documents(documents[i:i + batch_size])

    logger.info("✅ Successfully built combined Volve History & Geophysics DB.")

import re
if __name__ == "__main__":
    t0 = time.time()
    build_combined_db()
    logger.info(f"Total time: {time.time() - t0:.1f}s")
