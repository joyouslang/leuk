[Home](README.md) › Development

# Development

## Setup

```bash
uv sync --extra voice --group dev   # core + voice + dev tools
```

## Tests, lint, types

```bash
uv run pytest                       # full suite
uv run pytest tests/test_safety.py  # one file
uv run ruff check src/ tests/       # lint
uv run ruff format src/ tests/      # format
uv run mypy src/                    # type-check
```

Tests use `pytest-asyncio` (`asyncio_mode = "auto"`). `tests/conftest.py` provides
`MockProvider` and shared fixtures. Hardware/network-dependent paths (audio, ydotool,
live screenshots, real provider calls) are mocked.

## Code style

- Python 3.13+, `from __future__ import annotations` in every module.
- Line length 100; `ruff` for lint+format; `mypy` for types.
- Commits use [Conventional Commits](https://www.conventionalcommits.org/); author
  this project's commits as `joyouslang <rustofthedust@gmail.com>`; no
  `Co-Authored-By` lines.

## Adding a tool

1. Create `src/leuk/tools/<name>.py` implementing the `Tool` protocol
   (`spec` → `ToolSpec`, `async execute(arguments) -> str`).
2. Register it in `create_default_registry()` (`src/leuk/tools/__init__.py`),
   gated by a config toggle if it's optional/heavy.
3. Add a config block in `src/leuk/config.py` and, for high-risk tools, a
   [SafetyGuard](safety.md) rule.
4. Document it under [Tools](tools.md) (a dedicated `docs/tools/<name>.md` for
   non-trivial ones) and update the wiki index.

See [Browser](tools/browser.md) and [Input Control](tools/input_control.md) for
import-guarded optional-tool examples.

## Keeping docs in sync

This wiki is the [ground truth](README.md). Any behavior/config/command/tool/
provider/architecture change updates the matching page **in the same change**.

## See also

- [File Layout](reference/file-layout.md) · [Tools](tools.md) · [Architecture Overview](architecture.md)
