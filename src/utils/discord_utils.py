import asyncio
import logging

logger = logging.getLogger(__name__)

DISCORD_LIMIT = 2000
CHUNK_BUDGET = 1900  # leave room for prefix/formatting


def _split_text(text: str, budget: int = CHUNK_BUDGET) -> list[str]:
    """Split a string into chunks <= budget chars, preferring paragraph/sentence/word boundaries.

    Python string slicing is code-point based, so splitting at any index is safe for Thai UTF-8.
    This splitter just picks nicer boundaries so chunks don't land mid-sentence.
    """
    if len(text) <= budget:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > budget:
        window = remaining[:budget]
        # Prefer paragraph, then sentence-ish, then whitespace
        for sep in ("\n\n", "\n", "。", ". ", "! ", "? ", " "):
            idx = window.rfind(sep)
            if idx > budget // 2:  # avoid tiny chunks
                cut = idx + len(sep)
                chunks.append(remaining[:cut].rstrip())
                remaining = remaining[cut:].lstrip()
                break
        else:
            # No good boundary; hard-cut at budget
            chunks.append(remaining[:budget])
            remaining = remaining[budget:]
    if remaining:
        chunks.append(remaining)
    return chunks


async def send_long(ctx, text: str, prefix: str = "") -> None:
    """Send ``text`` to a Discord channel, chunking so each message respects the 2000-char limit.

    ``prefix`` is prepended to the first chunk only (e.g., a status marker).
    """
    if not text:
        if prefix:
            await ctx.send(prefix)
        return

    # Reserve room for the prefix on the first chunk.
    first_budget = CHUNK_BUDGET - len(prefix) - 1 if prefix else CHUNK_BUDGET
    first_budget = max(first_budget, 100)

    if len(text) <= first_budget:
        await ctx.send(f"{prefix}\n{text}" if prefix else text)
        return

    first = text[:first_budget]
    # Try to snap the first cut to a boundary too
    for sep in ("\n\n", "\n", ". ", " "):
        idx = first.rfind(sep)
        if idx > first_budget // 2:
            first = text[: idx + len(sep)]
            break
    rest = text[len(first):]

    await ctx.send(f"{prefix}\n{first.rstrip()}" if prefix else first.rstrip())
    for chunk in _split_text(rest):
        await ctx.send(chunk)


async def request_approval(bot, ctx, prompt_text: str | None = None, timeout: int = 300) -> bool:
    """Ask the user for ✅/❌ approval. Returns True iff they react ✅ in time.

    The caller is expected to have already broadcast the full content being approved
    (via ``send_long`` or similar). This function only posts a short prompt with
    reactions attached, so the approver has actually seen everything before deciding.

    Only the command's invoker can approve. Other users' reactions are ignored.
    """
    header = prompt_text or (
        "👨‍💼 **BOSS REVIEW**: Please read the full post above. "
        "React ✅ to approve and publish, or ❌ to reject."
    )
    msg = await ctx.send(header)
    await msg.add_reaction("✅")
    await msg.add_reaction("❌")

    def check(reaction, user) -> bool:
        return (
            user.id == ctx.author.id
            and reaction.message.id == msg.id
            and str(reaction.emoji) in ("✅", "❌")
        )

    try:
        reaction, _ = await bot.wait_for("reaction_add", timeout=timeout, check=check)
    except asyncio.TimeoutError:
        await ctx.send("⏱️ Approval timeout. Workflow cancelled.")
        return False

    if str(reaction.emoji) == "✅":
        return True
    await ctx.send("❌ Boss rejected. Content not posted.")
    return False
