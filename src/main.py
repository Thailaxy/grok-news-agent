import logging
import re
import unicodedata

import discord
from discord.ext import commands

from src.agents.editor import EditorAgent
from src.agents.engineer import EngineerAgent
from src.agents.image_gen import ImageAgent
from src.agents.writer import WriterAgent
from src.config import Config
from src.database import Database
from src.logging_setup import configure as configure_logging, context_logger
from src.utils.discord_utils import request_approval, send_long

# Validate config before anything else so missing tokens fail fast.
Config.validate()

configure_logging()
logger = logging.getLogger(__name__)

TOPIC_MIN = 2
TOPIC_MAX = 200


def _sanitize_topic(raw: str) -> str | None:
    """Strip control chars, collapse whitespace, enforce length limits.

    Returns the cleaned topic, or None if invalid.
    """
    if raw is None:
        return None
    # Drop control/format chars (keeps Thai — those are category Lo/Mn/Mc).
    cleaned = "".join(
        ch for ch in raw if unicodedata.category(ch)[0] not in ("C",)
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not (TOPIC_MIN <= len(cleaned) <= TOPIC_MAX):
        return None
    return cleaned


class SolarBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        super().__init__(command_prefix="!", intents=intents)

        # Initialize agents
        self.engineer = EngineerAgent()
        self.writer = WriterAgent()
        self.editor = EditorAgent()
        self.image_gen = ImageAgent()

        # Initialize database
        self.db = Database()

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)


bot = SolarBot()


@bot.command(name='solar')
async def solar(ctx, *, topic: str):
    """5-agent workflow: Boss → Engineer → Writer → Editor → Image Agent"""

    clean_topic = _sanitize_topic(topic)
    if clean_topic is None:
        await ctx.send(
            f"❌ Topic must be {TOPIC_MIN}-{TOPIC_MAX} characters after trimming. Try again."
        )
        return
    topic = clean_topic

    clog = context_logger(__name__, user_id=ctx.author.id, topic=topic)
    clog.info("solar workflow started")

    # Agent 1: BOSS - Initialize
    await ctx.send(f'👨‍💼 **BOSS INITIATED**: Processing "{topic}"...\n')

    try:
        # Agent 2: ENGINEER - Research
        await ctx.send(f'🔬 **ENGINEER**: Researching "{topic}"...')
        research_data = await bot.engineer.process(topic)
        if (
            not research_data
            or not research_data.get("raw_sources")
            or not (research_data.get("key_facts_th") or research_data.get("summary_th"))
        ):
            debug = (research_data or {}).get("debug") or "No details available."
            await ctx.send(
                "❌ Engineer couldn't gather usable research for that topic.\n"
                f"→ {debug}\n"
                "Try rephrasing with more specific keywords."
            )
            return

        facts = research_data.get("key_facts_th") or []
        facts_display = "\n".join(f"- {fact}" for fact in facts[:3]) or "- (ไม่มีข้อเท็จจริงที่สรุปได้)"
        await ctx.send(f'✅ Research complete. Key facts found:\n{facts_display}')

        # Agent 3: WRITER - Draft
        await ctx.send('✍️ **WRITER**: Drafting 700-word article...')
        article = await bot.writer.process(research_data)
        draft_preview = article if len(article) <= 800 else article[:800] + "…"
        await ctx.send(f'✅ Draft complete (preview):\n\n{draft_preview}')

        # Agent 4: EDITOR - Polish
        await ctx.send('🎨 **EDITOR**: Polishing for Facebook...')
        polished_post = await bot.editor.process(article)

        # Broadcast the full polished post BEFORE approval so the approver
        # actually reads every word, then attach reactions to a short prompt.
        await send_long(ctx, polished_post, prefix='📄 **EDITOR OUTPUT — review this before approving:**')
        approved = await request_approval(bot, ctx)
        user_id = str(ctx.author.id)
        if not approved:
            bot.db.log_post(topic, polished_post, research_data, user_id=user_id, approved=False)
            return

        # Agent 5: IMAGE AGENT - Generate prompt (only after approval)
        await ctx.send('🖼️ **IMAGE AGENT**: Creating image prompt...')
        image_prompt = await bot.image_gen.process(polished_post, topic)

        # Log to database
        bot.db.log_post(topic, polished_post, research_data, user_id=user_id, approved=True)

        # Final confirmation + image prompt (image prompt only visible after approval)
        await ctx.send('🚀 **APPROVED — ready to post.**')
        await ctx.send(f'**Image Prompt (for DALL-E):**\n```{image_prompt}```')

    except Exception as e:
        clog.error("Workflow failed: %s", e, exc_info=True)
        await ctx.send(f"❌ **Error during workflow:** {str(e)}")


if __name__ == "__main__":
    bot.run(Config.DISCORD_TOKEN)
