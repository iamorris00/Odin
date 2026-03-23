import logging
logging.basicConfig(level=logging.ERROR)
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
db = Chroma(persist_directory="data/viking_context/chroma_fallback", embedding_function=emb)
count = db._collection.count()
print(f"Total embedded chunks in DB: {count}")
