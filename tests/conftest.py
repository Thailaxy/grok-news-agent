"""Test config: ensure Config doesn't fail validation and agents don't hit real APIs."""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src...` works when pytest is run from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub env vars before src.config loads them.
os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("DEFAULT_LLM", "groq")
