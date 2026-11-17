"""API key loading."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

PROVIDER_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
}

_loaded = False


def load_env(path: Path = ENV_FILE, override: bool = False) -> list[str]:
    """Read `.env` into the process environment."""
    global _loaded
    _loaded = True
    if not path.is_file():
        return []

    applied: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    return applied


def ensure_loaded() -> None:
    """Load `.env` once per process."""
    if not _loaded:
        load_env()


def key_for(provider: str) -> str | None:
    """The configured key for one provider, or None."""
    ensure_loaded()
    for name in PROVIDER_ENV.get(provider, ()):
        value = os.environ.get(name)
        if value:
            return value
    return None


def require_key(provider: str) -> None:
    """Raise if no key is configured for the provider."""
    if provider not in PROVIDER_ENV:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(PROVIDER_ENV)}.")
    if key_for(provider) is not None:
        return

    names = " or ".join(PROVIDER_ENV[provider])
    raise RuntimeError(
        f"No API key for provider {provider!r}.\n"
        f"  Set {names} in {ENV_FILE} (see .env.example), or export it:\n"
        f"      export {PROVIDER_ENV[provider][0]}=..."
    )


def configured_providers() -> dict[str, bool]:
    """Which providers have a key."""
    ensure_loaded()
    return {name: key_for(name) is not None for name in PROVIDER_ENV}


def status() -> str:
    """Key status for printing."""
    ensure_loaded()
    lines = [f"env file: {ENV_FILE}"
             f"{'' if ENV_FILE.is_file() else '  (not found — copy .env.example)'}"]
    for name, ok in configured_providers().items():
        vars_ = "/".join(PROVIDER_ENV[name])
        lines.append(f"  {'set    ' if ok else 'missing'}  {name:<10} {vars_}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(status())
