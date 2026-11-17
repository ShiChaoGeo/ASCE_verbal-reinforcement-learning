"""On-disk record of one episode: state, action, rewards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "vrl_runs"

STATE_FILE = "state_cards.json"
ACTION_FILE = "locked_action.json"
VERBAL_FILE = "verbal_reward.json"
REFLECTION_FILE = "reflection.md"
META_FILE = "episode_meta.json"


def _read(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict | list) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Trajectory:
    """One (cpt_id, policy_version) episode record."""

    cpt_id: int | str
    policy_version: str

    @property
    def root(self) -> Path:
        return RUNS_DIR / f"cpt_{self.cpt_id}" / self.policy_version

    def path(self, filename: str) -> Path:
        return self.root / filename

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def save_state(self, cards: list[dict]) -> Path:
        """Save the feature cards the policy saw."""
        self.ensure()
        return _write(self.path(STATE_FILE), cards)

    def load_state(self) -> list[dict]:
        return _read(self.path(STATE_FILE))

    def save_action(self, action: list[dict], n_segments: int | None = None) -> Path:
        """Lock the action."""
        if n_segments is not None and len(action) != n_segments:
            raise ValueError(
                f"Expected {n_segments} segment actions, got {len(action)}. "
                "A partial action cannot be locked."
            )
        self.ensure()
        path = self.path(ACTION_FILE)
        if path.is_file():
            raise FileExistsError(
                f"{path} already exists. A locked action is immutable — "
                "re-locking would invalidate the out-of-sample reward."
            )
        return _write(
            path,
            {
                "cpt_id": self.cpt_id,
                "policy_version": self.policy_version,
                "status": "locked",
                "action": action,
            },
        )

    def load_action(self) -> list[dict]:
        data = _read(self.path(ACTION_FILE))
        return sorted(data["action"], key=lambda a: int(a["segment"]))

    @property
    def is_locked(self) -> bool:
        return self.path(ACTION_FILE).is_file()

    def save_verbal_reward(self, payload: dict) -> Path:
        """Record the verbal reward produced by self-reflection."""
        self.ensure()
        path = self.path(VERBAL_FILE)
        if path.is_file():
            raise FileExistsError(
                f"{path} already exists; delete it to redo the reflection."
            )
        return _write(path, payload)

    def load_verbal_reward(self) -> dict:
        return _read(self.path(VERBAL_FILE))

    @property
    def has_verbal_reward(self) -> bool:
        return self.path(VERBAL_FILE).is_file()

    def save_reflection(self, markdown: str) -> Path:
        """Readable record of the reflection."""
        self.ensure()
        path = self.path(REFLECTION_FILE)
        if path.is_file():
            raise FileExistsError(
                f"{path} already exists; delete it to redo the reflection."
            )
        path.write_text(markdown, encoding="utf-8")
        return path

    def save_meta(self, payload: dict) -> Path:
        self.ensure()
        return _write(self.path(META_FILE), payload)

    def __repr__(self) -> str:
        stage = "locked" if self.is_locked else "open"
        if self.has_verbal_reward:
            stage = "reflected"
        return f"Trajectory(cpt={self.cpt_id}, policy={self.policy_version}, {stage})"
