import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Default LLM to use: "gemini" or "groq"
    DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gemini")
    
    # Model names
    GEMINI_MODEL = "gemini-1.5-flash" # or gemini-1.5-pro
    GROQ_MODEL = "llama-3.3-70b-versatile"
