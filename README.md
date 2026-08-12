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
- **Junie** (JetBrains)
- **Kiro** (AWS)
- **Pi**
- **GitHub Copilot**
- **Replit**
- **Custom agents** (via the `AI_AGENT` environment variable)

See [`agents.json`](https://raw.githubusercontent.com/vercel/detect-agent/main/agents.json) for the exact environment variables and conditions used to detect each one.

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
