import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.engineer import EngineerAgent, ResearchData
from src.agents.writer import WriterAgent


# -- Engineer._parse_json ------------------------------------------------------

def test_engineer_parse_json_plain():
    out = EngineerAgent._parse_json(
        '{"overview_th":"สรุป","technical_th":["a","b"],"cost_roi_th":[],"faq_th":[]}'
    )
    assert out["overview_th"] == "สรุป"
    assert out["technical_th"] == ["a", "b"]


def test_engineer_parse_json_with_markdown_fence():
    raw = '```json\n{"overview_th":"x","technical_th":["a"]}\n```'
    out = EngineerAgent._parse_json(raw)
    assert out["overview_th"] == "x"


def test_engineer_parse_json_embedded_in_prose():
    raw = 'Here is the result:\n{"overview_th":"y"}\nThanks!'
    out = EngineerAgent._parse_json(raw)
    assert out["overview_th"] == "y"


def test_engineer_parse_json_invalid_returns_empty():
    assert EngineerAgent._parse_json("totally not json") == {}


def test_engineer_parse_json_rejects_list_top_level():
    assert EngineerAgent._parse_json('["a","b"]') == {}


# -- Engineer.process ----------------------------------------------------------

@pytest.mark.asyncio
async def test_engineer_process_empty_results_short_circuits():
    agent = EngineerAgent()
    with patch.object(agent, "_search", return_value=[]):
        agent.chat = AsyncMock(side_effect=AssertionError("chat should not be called"))
        out = await agent.process("topic-x")

    assert out["raw_sources"] == []
    assert out["overview_th"] == ""
    assert out["technical_th"] == []
    assert out["topic"] == "topic-x"
    assert "0 results" in out.get("debug", "")


@pytest.mark.asyncio
async def test_engineer_process_extracts_structured_sections():
    agent = EngineerAgent()
    fake_results = [
        {"title": "Thai solar ROI jumps", "body": "Savings data", "url": "https://x"},
        {"title": "New panels available", "body": "More data", "url": "https://y"},
    ]
    with patch.object(agent, "_search", return_value=fake_results):
        agent.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "overview_th": "โซลาร์เซลล์ในไทย",
                    "technical_th": ["แผงใช้ PV cell", "ใช้ inverter แปลงไฟ DC เป็น AC"],
                    "cost_roi_th": ["ROI ประมาณ 5-7 ปี"],
                    "faq_th": [{"q": "ติดตั้งยากไหม", "a": "ใช้เวลา 1-2 วัน"}],
                }
            )
        )
        out = await agent.process("solar roi")

    assert out["overview_th"] == "โซลาร์เซลล์ในไทย"
    assert out["technical_th"] == ["แผงใช้ PV cell", "ใช้ inverter แปลงไฟ DC เป็น AC"]
    assert out["cost_roi_th"] == ["ROI ประมาณ 5-7 ปี"]
    assert out["faq_th"] == [{"q": "ติดตั้งยากไหม", "a": "ใช้เวลา 1-2 วัน"}]
    assert len(out["raw_sources"]) == 2
    agent.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_engineer_process_filters_cjk_contaminated_facts():
    agent = EngineerAgent()
    with patch.object(agent, "_search", return_value=[{"title": "t", "body": "b", "url": "u"}]):
        agent.chat = AsyncMock(
            return_value=json.dumps(
                {
                    "overview_th": "โซลาร์",  # clean
                    "technical_th": [
                        "แผง PV ทำงานโดย 太陽能板",  # CJK — drop
                        "inverter แปลงไฟ DC เป็น AC",  # clean — keep
                    ],
                    "cost_roi_th": [],
                    "faq_th": [],
                }
            )
        )
        out = await agent.process("solar")

    assert out["technical_th"] == ["inverter แปลงไฟ DC เป็น AC"]
    assert "แผง PV ทำงานโดย 太陽能板" not in out["technical_th"]


@pytest.mark.asyncio
async def test_engineer_process_fails_when_all_sections_empty():
    agent = EngineerAgent()
    with patch.object(agent, "_search", return_value=[{"title": "t", "body": "b", "url": "u"}]):
        agent.chat = AsyncMock(
            return_value=json.dumps(
                {"overview_th": "", "technical_th": [], "cost_roi_th": [], "faq_th": []}
            )
        )
        out = await agent.process("solar")

    # Failure path clears raw_sources and sets debug.
    assert out["raw_sources"] == []
    assert "debug" in out and out["debug"]


# -- Writer --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writer_uses_structured_research_brief():
    agent = WriterAgent()
    agent.chat = AsyncMock(return_value="บทความเพื่อการศึกษา")
    research: ResearchData = {
        "topic": "on-grid vs off-grid",
        "overview_th": "ระบบ on-grid กับ off-grid ต่างกัน",
        "technical_th": ["on-grid เชื่อมต่อกับสายส่ง", "off-grid ใช้ battery"],
        "cost_roi_th": ["on-grid ROI 5 ปี"],
        "faq_th": [{"q": "เลือกแบบไหน", "a": "ขึ้นอยู่กับพื้นที่"}],
        "raw_sources": [],
    }
    result = await agent.process(research)

    assert result == "บทความเพื่อการศึกษา"
    call_args = agent.chat.await_args
    assert call_args is not None
    prompt = call_args.args[0]
    assert "ระบบ on-grid กับ off-grid ต่างกัน" in prompt
    assert "on-grid เชื่อมต่อกับสายส่ง" in prompt
    assert "on-grid ROI 5 ปี" in prompt
    assert "เลือกแบบไหน" in prompt
    assert "on-grid vs off-grid" in prompt


@pytest.mark.asyncio
async def test_writer_handles_missing_sections():
    """Writer should still render its prompt cleanly when only overview is present."""
    agent = WriterAgent()
    agent.chat = AsyncMock(return_value="บทความสั้น")
    research: ResearchData = {
        "topic": "solar basics",
        "overview_th": "โซลาร์ช่วยประหยัดพลังงาน",
        "technical_th": [],
        "cost_roi_th": [],
        "faq_th": [],
        "raw_sources": [],
    }
    result = await agent.process(research)
    assert result == "บทความสั้น"
    prompt = agent.chat.await_args.args[0]
    assert "โซลาร์ช่วยประหยัดพลังงาน" in prompt
    # Empty sections render as "(none)" rather than crashing.
    assert "(none)" in prompt
