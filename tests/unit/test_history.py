"""Unit tests for ConversationHistory and HistoryStore."""

from __future__ import annotations

import pytest

from scinanoai.chatbot.core.history import ConversationHistory, HistoryStore
from scinanoai.chatbot.llm.base import LLMMessage


@pytest.mark.unit
def test_append_and_snapshot_are_isolated() -> None:
    history = ConversationHistory(token_budget=4096)
    history.append(LLMMessage(role="user", content="hello"))
    snapshot = history.snapshot()
    snapshot.append(LLMMessage(role="user", content="mutation"))
    assert len(history.snapshot()) == 1


@pytest.mark.unit
def test_limit_to_budget_drops_oldest() -> None:
    history = ConversationHistory(token_budget=20)
    messages = [LLMMessage(role="user", content="x" * 50) for _ in range(5)]
    limited = history.limit_to_budget(messages, budget=20)
    assert len(limited) <= len(messages)
    assert limited == messages[-len(limited) :]


@pytest.mark.unit
def test_summarize_skipped_when_below_budget() -> None:
    history = ConversationHistory(token_budget=1_000_000)
    history.append(LLMMessage(role="user", content="hi"))
    history.summarize_if_needed(lambda _msgs: "SUMMARY")
    assert [m.content for m in history.snapshot()] == ["hi"]


@pytest.mark.unit
def test_summarize_replaces_old_messages_when_over_budget() -> None:
    history = ConversationHistory(token_budget=1, summarize_after_messages=2)
    for i in range(6):
        history.append(LLMMessage(role="user", content=f"msg-{i}"))

    history.summarize_if_needed(lambda _msgs: "SUMMARY")

    snapshot = history.snapshot()
    assert snapshot[0].role == "system"
    assert "SUMMARY" in snapshot[0].content
    assert [m.content for m in snapshot[-2:]] == ["msg-4", "msg-5"]


@pytest.mark.unit
def test_history_store_isolates_sessions() -> None:
    store = HistoryStore(token_budget=4096)
    a = store.get("alice")
    b = store.get("bob")
    a.append(LLMMessage(role="user", content="alice-question"))
    assert b.snapshot() == []
    assert len(a.snapshot()) == 1


@pytest.mark.unit
def test_history_store_clear() -> None:
    store = HistoryStore(token_budget=4096)
    history = store.get("u1")
    history.append(LLMMessage(role="user", content="x"))
    store.clear("u1")
    assert store.get("u1").snapshot() == []
