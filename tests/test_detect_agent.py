"""Tests for determine_agent, driven primarily by upstream testcases.json."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import detect_agent
from detect_agent import KNOWN_AGENTS, determine_agent
from detect_agent import _evaluate as evaluate_mod

_TESTCASES_PATH = Path(__file__).with_name("testcases.json")
_TESTCASES: list[dict[str, Any]] = json.loads(_TESTCASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def agent_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    # Never inspect the host environment: every detector reads this synthetic mapping.
    env = {"PATH": "/usr/bin"}
    fake_os = SimpleNamespace(environ=env)
    monkeypatch.setattr(detect_agent, "os", fake_os)
    monkeypatch.setattr(evaluate_mod, "os", fake_os)
    monkeypatch.setattr(evaluate_mod, "is_tty_fn", lambda: True)
    monkeypatch.setattr(evaluate_mod, "path_exists_fn", lambda _path: False)
    return env


def _expected_name(tc: dict[str, Any]) -> str | None:
    if "expectedName" in tc:
        return tc["expectedName"]
    key = tc.get("expectedAgentKey")
    if key is None:
        return None
    return KNOWN_AGENTS[key]


@pytest.mark.parametrize("tc", _TESTCASES, ids=[tc["name"] for tc in _TESTCASES])
def test_upstream_testcase(
    tc: dict[str, Any], monkeypatch: pytest.MonkeyPatch, agent_env: dict[str, str]
):
    agent_env.update(tc.get("env", {}))

    if tc.get("tty") is not None:
        monkeypatch.setattr(evaluate_mod, "is_tty_fn", lambda: bool(tc["tty"]))

    files = set(tc.get("files") or [])
    if files:
        monkeypatch.setattr(evaluate_mod, "path_exists_fn", lambda path: path in files)

    result = determine_agent()
    expected_name = _expected_name(tc)

    if tc["expectedIsAgent"]:
        assert result == {"is_agent": True, "agent": {"name": expected_name}}
    else:
        assert result == {"is_agent": False, "agent": None}


class TestPythonPortExtensions:
    """Behavior kept for Python-port compatibility / extra coverage."""

    def test_pi_coding_agent_set_detects_pi(self, agent_env: dict[str, str]):
        agent_env["PI_CODING_AGENT"] = "1"
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": KNOWN_AGENTS["PI"]}}

    def test_pi_coding_agent_keeps_priority_over_new_contexts(self, agent_env: dict[str, str]):
        agent_env.update({"PI_CODING_AGENT": "1", "CRUSH": "1", "ZED_TERM": "1"})
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": KNOWN_AGENTS["PI"]}}

    def test_ai_agent_github_copilot_cli_is_emitted_verbatim(self, agent_env: dict[str, str]):
        # Upstream emits AI_AGENT values verbatim (no alias rewriting).
        agent_env["AI_AGENT"] = "github-copilot-cli"
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": "github-copilot-cli"}}

    def test_ai_agent_v0(self, agent_env: dict[str, str]):
        agent_env["AI_AGENT"] = "v0"
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": "v0"}}


class TestNewUpstreamAgents:
    """Coverage for agents added upstream after the initial JSON testcases."""

    def test_cline_active_detects_cline(self, agent_env: dict[str, str]):
        agent_env["CLINE_ACTIVE"] = "1"
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": KNOWN_AGENTS["CLINE"]}}

    def test_openclaw_shell_detects_openclaw(self, agent_env: dict[str, str]):
        agent_env["OPENCLAW_SHELL"] = "1"
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": KNOWN_AGENTS["OPENCLAW"]}}

    def test_codex_sandbox_network_disabled_detects_codex(self, agent_env: dict[str, str]):
        agent_env["CODEX_SANDBOX_NETWORK_DISABLED"] = "1"
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": KNOWN_AGENTS["CODEX"]}}

    def test_antigravity_cli_alias_detects_antigravity(self, agent_env: dict[str, str]):
        agent_env["ANTIGRAVITY_CLI_ALIAS"] = "1"
        result = determine_agent()
        assert result == {"is_agent": True, "agent": {"name": KNOWN_AGENTS["ANTIGRAVITY"]}}


class TestEvaluateCondition:
    """Unit tests for condition leaves/combinators."""

    def test_env_set_true(self, agent_env: dict[str, str]):
        agent_env["SOME_VAR"] = "1"
        assert evaluate_mod.evaluate_condition({"type": "env_set", "name": "SOME_VAR"})

    def test_env_set_false_when_empty(self, agent_env: dict[str, str]):
        agent_env["SOME_VAR"] = ""
        assert not evaluate_mod.evaluate_condition({"type": "env_set", "name": "SOME_VAR"})

    def test_env_value_match(self, agent_env: dict[str, str]):
        agent_env["ROLE"] = "agent-exec"
        assert evaluate_mod.evaluate_condition(
            {"type": "env_value", "name": "ROLE", "value": "agent-exec"}
        )

    def test_env_matches_and_malformed_pattern(self, agent_env: dict[str, str]):
        agent_env["PATH"] = "/home/me/.pi/agent/bin"
        assert evaluate_mod.evaluate_condition(
            {"type": "env_matches", "name": "PATH", "pattern": r"\.pi[\\/]agent"}
        )
        assert not evaluate_mod.evaluate_condition(
            {"type": "env_matches", "name": "PATH", "pattern": "("}
        )

    def test_any_of_and_all_of(self, agent_env: dict[str, str]):
        agent_env["A"] = "1"
        assert evaluate_mod.evaluate_condition(
            {
                "type": "anyOf",
                "conditions": [
                    {"type": "env_set", "name": "MISSING"},
                    {"type": "env_set", "name": "A"},
                ],
            }
        )
        assert not evaluate_mod.evaluate_condition(
            {
                "type": "allOf",
                "conditions": [
                    {"type": "env_set", "name": "A"},
                    {"type": "env_set", "name": "MISSING"},
                ],
            }
        )

    def test_unknown_condition_type_is_false(self):
        assert not evaluate_mod.evaluate_condition({"type": "not-a-real-type"})
