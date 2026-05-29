# Changelog

## [0.3.0](https://github.com/togethercomputer/detect_agent/compare/detect_agent-v0.2.1...detect_agent-v0.3.0) (2026-05-29)


### Features

* Add v0 detection, improve cursor detection ([#9](https://github.com/togethercomputer/detect_agent/issues/9)) ([97e7e36](https://github.com/togethercomputer/detect_agent/commit/97e7e365e8600200d715259301ab40403a980d04))

## 0.2.1

- Add support for v0 via `AI_AGENT=v0`
- Sync Cursor detection with upstream: `CURSOR_TRACE_ID` detects Cursor IDE agent-terminal sessions, and `CURSOR_AGENT` detects cursor-cli commands

## 0.2.0

- Improve detection for cursor agents
- Add support for pi coding agent

## 0.1.1

- Improve detection for cursor usage to avoid mistaking humans as agents

## 0.1.0

Initial Release
