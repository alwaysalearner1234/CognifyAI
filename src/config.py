"""Central config for CognifyAI."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("KB_DIR", BASE_DIR / "data" / "kb"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads"))
KB_UNITS_FILE = KB_DIR / "units.json"
KB_INDEX_FILE = KB_DIR / "tfidf_index.pkl"
LEARNER_FILE = KB_DIR / "learner.json"
QUIZ_HISTORY_FILE = KB_DIR / "quiz_history.json"

KB_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # auto|gemini|openai|local
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
TOP_K = 5
REFUSAL_THRESHOLD = 0.10  # min TF-IDF cosine score to consider grounded

# BKT defaults
BKT_P_INIT = 0.30
BKT_P_TRANSIT = 0.15
BKT_P_GUESS = 0.20
BKT_P_SLIP = 0.10
MASTERY_THRESHOLD = 0.75
