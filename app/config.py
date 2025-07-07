import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_db"

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
AUTH = (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
