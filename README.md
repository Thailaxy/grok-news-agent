# Wanakorn Solar Agent (Multi-Agent AI)

A Discord bot that researches solar-energy topics for the Thai market and produces a Thai-language Facebook post plus a DALL-E image prompt, using a 5-agent pipeline. Powered by Groq (Llama 3.3) or Gemini (1.5 Flash) — both free tiers — and DuckDuckGo Search.

## Features

* **Engineer**: Searches Thai-region news via DuckDuckGo and synthesizes the findings into Thai-language key facts.
* **Writer**: Drafts a ~700-word Thai Facebook article (short paragraphs, sub-sections with emoji markers, one CTA) grounded only in the engineer's facts.
* **Editor**: Polishes the draft for Facebook (grammar, flow, emojis, hashtags).
* **Boss (approval)**: Requires ✅/❌ reaction from the command invoker before the final post and image prompt are released.
* **Image Agent**: Generates a DALL-E / Midjourney prompt text (prompt-only, no image API call).

Runs are logged to a local SQLite database (`posts_log.db`) with topic, user, timestamp, and approval status.

## Setup Instructions

1. Copy `.env.example` to `.env` and fill in your credentials:
   - `GROQ_API_KEY`: Your Groq API key
   - `DISCORD_TOKEN`: Your Discord bot token
   - `GEMINI_API_KEY`: Your Gemini API key (required if `DEFAULT_LLM=gemini`)
   - `DEFAULT_LLM`: `groq` or `gemini`
2. Install the required dependencies using `pip install -r requirements.txt`

## Usage

The active bot is the multi-agent solar workflow in `src/main.py`. Run with:

```
python -m src.main
```

Then in Discord:

```
!solar <topic>
```

Example: `!solar โซล่าเซลล์บ้าน`

The legacy single-file bot (`groq_agent.py`) exposes `!news <topic>` and is kept for reference.

## Development

Install dev deps and run tests:

```
pip install -r requirements-dev.txt
pytest -q
```
