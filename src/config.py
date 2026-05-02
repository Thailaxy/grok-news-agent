import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Default LLM to use: "gemini" or "groq"
    DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gemini").lower()

    # Model names
    GEMINI_MODEL = "gemini-1.5-flash"  # or gemini-1.5-pro
    GROQ_MODEL = "llama-3.3-70b-versatile"

    @classmethod
    def validate(cls) -> None:
        """Fail fast if required env vars are missing.

        Must be called at startup before the bot connects.
        """
        missing: list[str] = []
        if not cls.DISCORD_TOKEN:
            missing.append("DISCORD_TOKEN")

        if cls.DEFAULT_LLM not in ("gemini", "groq"):
            raise RuntimeError(
                f"DEFAULT_LLM must be 'gemini' or 'groq' (got {cls.DEFAULT_LLM!r})"
            )

        # Require the key for whichever provider is active,
        # and the other key only if it's likely to be used (we accept it missing).
        if cls.DEFAULT_LLM == "gemini" and not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if cls.DEFAULT_LLM == "groq" and not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )
