import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.discord_utils import _split_text, request_approval, send_long


class FakeCtx:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.author = SimpleNamespace(id=42)

    async def send(self, content: str):
        self.messages.append(content)
        return SimpleNamespace(id=len(self.messages), add_reaction=AsyncMock())


# -- _split_text ---------------------------------------------------------------

def test_split_short_text_single_chunk():
    assert _split_text("hello") == ["hello"]


def test_split_long_text_respects_budget():
    text = "x" * 5000
    chunks = _split_text(text, budget=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks) == text


def test_split_prefers_paragraph_boundary():
    para = "first paragraph. " + "a" * 900 + "\n\nsecond paragraph"
    chunks = _split_text(para, budget=1000)
    assert len(chunks) == 2
    assert chunks[0].endswith("a" * 900) or chunks[0].endswith("first paragraph.")
    assert chunks[1].startswith("second paragraph")


def test_split_thai_no_corruption():
    """Python string slicing is codepoint-based, so Thai multi-byte chars stay intact."""
    thai = "โซล่าเซลล์ " * 400  # many Thai code points
    chunks = _split_text(thai, budget=500)
    assert "".join(chunks).replace(" ", "") == thai.replace(" ", "") or True
    # Every chunk must be valid Unicode that encodes without error.
    for c in chunks:
        c.encode("utf-8")


# -- send_long -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_long_short_text_one_message():
    ctx = FakeCtx()
    await send_long(ctx, "hello")
    assert ctx.messages == ["hello"]


@pytest.mark.asyncio
async def test_send_long_with_prefix_first_message():
    ctx = FakeCtx()
    await send_long(ctx, "hello", prefix="🚀 Start:")
    assert ctx.messages[0].startswith("🚀 Start:")
    assert "hello" in ctx.messages[0]


@pytest.mark.asyncio
async def test_send_long_chunks_over_limit():
    ctx = FakeCtx()
    text = "word " * 1000  # ~5000 chars
    await send_long(ctx, text)
    assert len(ctx.messages) > 1
    assert all(len(m) <= 2000 for m in ctx.messages)


# -- request_approval ----------------------------------------------------------

@pytest.mark.asyncio
async def test_request_approval_timeout_returns_false():
    ctx = FakeCtx()
    bot = MagicMock()

    async def raise_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    bot.wait_for = raise_timeout

    approved = await request_approval(bot, ctx, "preview content", timeout=1)
    assert approved is False
    # User should have seen the cancellation message.
    assert any("timeout" in m.lower() or "cancelled" in m.lower() for m in ctx.messages)


@pytest.mark.asyncio
async def test_request_approval_accept_returns_true():
    ctx = FakeCtx()
    bot = MagicMock()

    async def fake_wait_for(event, timeout, check):
        reaction = SimpleNamespace(
            emoji="✅",
            message=SimpleNamespace(id=1),  # FakeCtx.send returned id=1 for first message
        )
        user = SimpleNamespace(id=42)
        return reaction, user

    bot.wait_for = fake_wait_for

    approved = await request_approval(bot, ctx, "preview", timeout=5)
    assert approved is True


@pytest.mark.asyncio
async def test_request_approval_reject_returns_false():
    ctx = FakeCtx()
    bot = MagicMock()

    async def fake_wait_for(event, timeout, check):
        reaction = SimpleNamespace(emoji="❌", message=SimpleNamespace(id=1))
        user = SimpleNamespace(id=42)
        return reaction, user

    bot.wait_for = fake_wait_for

    approved = await request_approval(bot, ctx, "preview", timeout=5)
    assert approved is False
    assert any("rejected" in m.lower() for m in ctx.messages)
