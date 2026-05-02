import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.engineer import EngineerAgent, ResearchData
from src.agents.writer import WriterAgent


# -- Engineer ------------------------------------------------------------------

def test_engineer_parse_json_plain():
    out = EngineerAgent._parse_json(
        '{"summary_th":"สรุป","key_facts_th":["a","b"]}'
    )
    assert out == {"summary_th": "สรุป", "key_facts_th": ["a", "b"]}


def test_engineer_parse_json_with_markdown_fence():
    raw = '```json\n{"summary_th":"x","key_facts_th":["a"]}\n```'
    out = EngineerAgent._parse_json(raw)
    assert out["summary_th"] == "x"
    assert out["key_facts_th"] == ["a"]


def test_engineer_parse_json_embedded_in_prose():
    raw = 'Here is the result:\n{"summary_th":"y","key_facts_th":["b"]}\nThanks!'
    out = EngineerAgent._parse_json(raw)
    assert out["summary_th"] == "y"


def test_engineer_parse_json_invalid_returns_empty():
    assert EngineerAgent._parse_json("totally not json") == {}


def test_engineer_parse_json_rejects_list_top_level():
    assert EngineerAgent._parse_json('["a","b"]') == {}


@pytest.mark.asyncio
async def test_engineer_process_empty_results_short_circuits():
    agent = EngineerAgent()
    with patch.object(agent, "_search", return_value=[]):
        # chat should never be called since we short-circuit.
        agent.chat = AsyncMock(side_effect=AssertionError("chat should not be called"))
        out = await agent.process("topic-x")

    assert out["raw_sources"] == []
    assert out["key_facts_th"] == []
    assert out["topic"] == "topic-x"


@pytest.mark.asyncio
async def test_engineer_process_synthesizes_thai_facts():
    agent = EngineerAgent()
    fake_results = [
        {"title": "Thai solar ROI jumps", "body": "Data about savings", "url": "https://x"},
        {"title": "New panels available", "body": "More data", "url": "https://y"},
    ]
    with patch.object(agent, "_search", return_value=fake_results):
        agent.chat = AsyncMock(
            return_value=json.dumps(
                {"summary_th": "สรุปการลงทุน", "key_facts_th": ["ข้อ1", "ข้อ2"]}
            )
        )
        out = await agent.process("solar roi")

    assert out["summary_th"] == "สรุปการลงทุน"
    assert out["key_facts_th"] == ["ข้อ1", "ข้อ2"]
    assert len(out["raw_sources"]) == 2
    agent.chat.assert_awaited_once()


# -- Writer --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writer_uses_thai_facts_not_raw_news():
    agent = WriterAgent()
    agent.chat = AsyncMock(return_value="บทความภาษาไทย 500 คำ")
    research: ResearchData = {
        "topic": "โซล่าเซลล์",
        "summary_th": "สรุปภาษาไทย",
        "key_facts_th": ["ข้อเท็จจริง 1", "ข้อเท็จจริง 2"],
        "raw_sources": [],
    }
    result = await agent.process(research)

    assert result == "บทความภาษาไทย 500 คำ"
    # Verify the prompt was built from Thai summary + facts, not any English raw_news.
    call_args = agent.chat.await_args
    assert call_args is not None
    prompt = call_args.args[0]
    assert "สรุปภาษาไทย" in prompt
    assert "ข้อเท็จจริง 1" in prompt
    assert "โซล่าเซลล์" in prompt
