# detect_agent

> This is a Python port of Vercel's [`detect-agent`](https://github.com/vercel/detect-agent) package.

A lightweight utility for detecting if code is being executed by an AI agent or automated development environment.

## Installation

```bash
uv add detect_agent
```

## Usage

```python
from detect_agent import determine_agent

result = determine_agent()

if result["is_agent"]:
    agent = result["agent"]
    print(f"Running in {agent['name']} environment")
    # Adapt behavior for AI agent context
```

## Agent Definitions

Detection rules are vendored from upstream [`agents.json`](https://raw.githubusercontent.com/vercel/detect-agent/main/agents.json) (validated against [`agents.schema.json`](https://raw.githubusercontent.com/vercel/detect-agent/main/agents.schema.json)) and evaluated in array order; the first match wins.

## Supported Agents

This package can detect the following AI agents and development environments:

- **Cursor** (Anysphere)
- **Claude Code** (Anthropic)
- **Claude Cowork** (Anthropic)
- **Kimi Code** (Moonshot AI)
- **Grok Build** (xAI)
- **Devin** (Cognition Labs)
- **Gemini CLI** (Google)
- **Codex** (OpenAI)
- **Antigravity** (Google DeepMind)
- **Augment**
- **Cline**
- **OpenCode**
- **OpenClaw**
- **Goose** (Block)
- **Junie** (JetBrains; non-empty `JUNIE_SHIM_PATH` only; `JUNIE_DATA` is ignored)
- **Kiro** (AWS)
- **Pi**
- **GitHub Copilot**
- **Replit**
- **Custom agents** (via the `AI_AGENT` environment variable)
- **Crush** (Charm)
- **Hermes Agent** (Nous Research)
- **Kilo Code**
- **TRAE**
- **VT Code**
- **Zed**
- **WorkBuddy**
- **Grok Bot** (separate from Grok Build)
- **[QwenWork](https://qwenwork.cn)**
- **Cue**
- **Muse**
- **Manus**
- **Doubao Work**

See the bundled [`agents.json`](detect_agent/agents.json) for the exact environment variables and conditions used to detect each one.

### Environment context and rule sources

These rules identify environment contexts; a match does not authenticate the process that
started a command. In particular, `ZED_TERM` identifies a Zed terminal, including a
human-operated terminal, so it is a fallback after other supported contexts.

Crush (`CRUSH`), Hermes (`HERMES_SESSION_ID`), Kilo Code (`KILOCODE_FEATURE`),
Trae (`TRAE_AI_SHELL_ID`), VTCode (`VTCODE=1`), and Zed (`ZED_TERM`) markers are
also registered in the [Hugging Face agent-harness registry at commit
`e082407`](https://github.com/huggingface/huggingface.js/blob/e08240742cd64d8f908b22e52358b84c812e882c/packages/tasks/src/agent-harnesses.ts).
The same registry provides `GOOSE_TERMINAL` and `AGENT_CONTEXT_OUT`, which extend
the existing Goose and Kiro rules without removing their previous conditions.
Hermes additionally [advertises
`HERMES_AGENT`](https://github.com/NousResearch/hermes-agent/blob/e5eb6b36fe78eb012cefd1f5f436cd458411b8b3/hermes_cli/main.py#L3268).
Among these registry-derived additions, VTCode requires the exact value `1`; the other
markers require a non-empty value.

WorkBuddy uses exact `WORKBUDDY_APPLICATION_NAME=WorkBuddy` or `WORKBUDDY_APP_NAME=WorkBuddy` markers;
generic packaging or configuration variables are insufficient. Grok Bot uses
`SAND_AGENT=1`, or a cloud-shell combination of non-empty `CURSOR_AGENT` with
exact `SAND_MANIFEST_V2=1` and `SAND_SUPERVISOR_ENABLED=1`, ahead of the Cursor
compatibility rules. Grok Build
remains a separate result and additionally accepts the exact terminal marker
`GROK_AGENT=1`. QwenWork checks for a non-empty
`QODER_WORK_INTEGRATION_PRODUCT`, returning `qwenwork`. The existing `env_set`
semantics accept any non-empty value and reject missing or empty values. The
inspected build sets `qwenworkcn`, but the rule does not restrict the value;
shared use of this variable by other products has not been established. Session
IDs, region/site flags, execution mode and service endpoints do not independently
identify the product. See the [marker evidence notes](docs/environment-marker-evidence.md)
for official product descriptions, inspected versions, artifact hashes, propagation
paths and the Grok Build source.
Cue uses the exact `MANUS_LONG_DESKTOP_WORKER=1` marker from a user-provided
Linux desktop-worker runtime observation. Despite the variable prefix, this rule
identifies Cue, not Manus. Muse uses exact `JARVIS_IS_ASSIGNED=1` or a non-empty
`JARVIS_TOOL_CALL_ID` from user-provided active-agent and tool-call observations.
Other configuration, instance, session, tier or location variables do not
independently trigger these rules.
The Manus rule checks a non-empty `MANUS_ADDON_RUNTIME_SPEC_PATH`
or `MANUS_ADDON_RUNTIME_SPECS_REVISION`. These descriptor names were reported in
an ordinary shell; their values were not supplied. The Grok Bot cloud-shell
combination also comes from a user observation. Shared infrastructure markers
do not establish exclusive product ownership or the immediate caller.
Doubao Work checks whether `DOUBAO_OFFICE_AGENT_NAME` has a non-empty value,
using the existing `env_set` semantics. Its value is not compared to a fixed
agent name; missing and empty values do not match. This variable comes from a
user-provided runtime observation. Other message, application, proxy, credential
and deployment fields do not independently identify it.

New agent-specific markers precede inherited Cursor terminal context. The relative
order and identifiers of existing rules are preserved, as are the `AI_AGENT` override
and the Python port's `PI_CODING_AGENT` compatibility check.
Harnesses such as `hi` and `sandbase-harness` can already identify themselves with
`AI_AGENT`; they do not need dedicated variable rules.

## The AI_AGENT Standard

We're promoting `AI_AGENT` as a universal environment variable standard for AI development tools. This allows any tool or library to easily detect when it's running in an AI-driven environment.

### For AI Tool Developers

Set the `AI_AGENT` environment variable to identify your tool:

```bash
export AI_AGENT="your-tool-name"
# or
AI_AGENT="your-tool-name" your-command
```

### Recommended Naming Convention

- Use lowercase with hyphens for multi-word names
- Include version information if needed, separated by an `@` symbol
- Examples: `claude-code`, `cursor-cli`, `devin@1`, `custom-agent@2.0`

## Development

```bash
uv sync --extra dev
uv run pytest
# Lint and format check (CI)
uv run ruff check . && uv run ruff format --check .
# Fix and format
uv run ruff check . --fix && uv run ruff format .
```

## Use Cases

### Adaptive Behavior

```python
import os

from detect_agent import determine_agent


def setup_environment():
    result = determine_agent()

    if result["is_agent"]:
        agent = result["agent"]
        # Running in AI environment - adjust behavior
        os.environ["LOG_LEVEL"] = "verbose"
        print(f"Detected AI agent: {agent['name']}")
```

### Telemetry and Analytics

```python
import time

from detect_agent import determine_agent


def track_usage(event: str):
    result = determine_agent()

    analytics.track(
        event,
        {
            "agent": result["agent"]["name"] if result["is_agent"] else "human",
            "timestamp": time.time(),
        },
    )
```

### Feature Toggles

```python
from detect_agent import determine_agent


def should_enable_feature(feature: str) -> bool:
    result = determine_agent()

    # Enable experimental features for AI agents
    if result["is_agent"] and feature == "experimental-ai-mode":
        return True

    return False
```

## Contributing

### Adding New Agent Support

To add support for a new AI agent:

1. Sync detection rules from upstream [`agents.json`](https://github.com/vercel/detect-agent/blob/main/agents.json) into `detect_agent/agents.json`. Agents are evaluated in array order — the first match wins.
2. Sync or extend test cases in `tests/testcases.json`.
3. Update this README with the new agent information.

## Links

- [GitHub Repository](https://github.com/togethercomputer/detect_agent)
- [Vercel upstream package](https://github.com/vercel/detect-agent)
