# detect_agent

> This is a Python port of Vercel's `@vercel/detect-agent` npm package.

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

## Supported Agents

This package can detect the following AI agents and development environments:

- **Custom agents** via `AI_AGENT` environment variable
- **Cursor** (cursor editor and cursor-cli)
- **Claude Code** (Anthropic's Claude)
- **Devin** (Cognition Labs)
- **Gemini CLI** (Google)
- **Codex** (OpenAI)
- **Antigravity** (Google DeepMind)
- **GitHub Copilot** (via `AI_AGENT=github-copilot|github-copilot-cli`, `COPILOT_MODEL`, `COPILOT_ALLOW_ALL`, or `COPILOT_GITHUB_TOKEN`)
- **Replit** (online IDE)
- **v0** (Vercel's AI assistant, via `AI_AGENT=v0`)

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

1. Add detection logic to `detect_agent/__init__.py`
2. Add comprehensive test cases in `tests/test_detect_agent.py`
3. Update this README with the new agent information
4. Follow the existing priority order pattern

## Links

- [GitHub Repository](https://github.com/togethercomputer/detect_agent)
- [Vercel upstream package](https://github.com/vercel/vercel/tree/main/packages/detect-agent)
