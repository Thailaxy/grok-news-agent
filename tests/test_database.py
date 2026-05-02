import json

from src.database import Database


def test_log_and_recent_roundtrip(tmp_path):
    db = Database(file_path=str(tmp_path / "test.db"))
    rid = db.log_post(
        topic="โซล่าเซลล์บ้าน",
        post="สวัสดี",
        research_data={"key_facts_th": ["a", "b"]},
        user_id="user-1",
        approved=True,
    )
    assert rid > 0

    rows = db.recent(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert row["topic"] == "โซล่าเซลล์บ้าน"
    assert row["post"] == "สวัสดี"
    assert row["user_id"] == "user-1"
    assert row["approved"] == 1


def test_log_rejected(tmp_path):
    db = Database(file_path=str(tmp_path / "test.db"))
    db.log_post("topic", "post", {"x": 1}, user_id="u", approved=False)
    rows = db.recent()
    assert rows[0]["approved"] == 0


def test_recent_order_newest_first(tmp_path):
    db = Database(file_path=str(tmp_path / "test.db"))
    db.log_post("first", "p", None)
    db.log_post("second", "p", None)
    db.log_post("third", "p", None)
    rows = db.recent(limit=10)
    assert [r["topic"] for r in rows] == ["third", "second", "first"]


def test_research_json_preserves_thai(tmp_path):
    db = Database(file_path=str(tmp_path / "test.db"))
    data = {"summary_th": "สรุปภาษาไทย", "key_facts_th": ["ข้อเท็จจริง"]}
    db.log_post("t", "p", data)
    # Read back via the connection directly.
    cur = db._conn.execute("SELECT research_json FROM posts LIMIT 1")
    stored = cur.fetchone()[0]
    loaded = json.loads(stored)
    assert loaded == data


def test_concurrent_writes_all_persist(tmp_path):
    """TOCTOU regression: the old JSON impl lost writes under concurrency."""
    import threading

    db = Database(file_path=str(tmp_path / "test.db"))

    def write(i: int) -> None:
        db.log_post(f"topic-{i}", "p", None)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = db.recent(limit=100)
    assert len(rows) == 20
