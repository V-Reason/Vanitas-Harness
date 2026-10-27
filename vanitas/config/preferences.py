"""跨会话偏好：只保留 Preferences 数据类与磁盘读写，不依赖 schema。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..errors import ConfigError

PREFERENCES_VERSION = 1


@dataclass
class Preferences:
    """跨会话偏好；损坏或缺失时回落到空状态，不阻断启动。"""

    path: Path = field(
        default_factory=lambda: Path.home() / ".vanitas" / "preferences.json"
    )
    last_used_model: str = ""
    preferred_profiles: dict[str, str] = field(default_factory=dict)
    last_updated: str = ""

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Preferences":
        target = (
            Path(path).expanduser()
            if path is not None
            else Path.home() / ".vanitas" / "preferences.json"
        )
        empty = cls(path=target)
        if not target.is_file():
            return empty
        try:
            raw = target.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(data, dict) or data.get("version") != PREFERENCES_VERSION:
            return empty
        last_model = data.get("last_used_model", "")
        preferred: dict[str, str] = {}
        raw_preferred = data.get("preferred_profiles")
        if isinstance(raw_preferred, dict):
            for model_id, key in raw_preferred.items():
                if (
                    isinstance(model_id, str)
                    and isinstance(key, str)
                    and model_id.strip()
                    and key.strip()
                ):
                    preferred[model_id.strip()] = key.strip()
        updated = data.get("last_updated", "")
        return cls(
            path=target,
            last_used_model=last_model.strip() if isinstance(last_model, str) else "",
            preferred_profiles=preferred,
            last_updated=updated if isinstance(updated, str) else "",
        )

    def profile_for(self, model_id: str) -> str | None:
        return self.preferred_profiles.get(model_id)

    def set_profile(self, model_id: str, profile_key: str) -> Path:
        self.preferred_profiles[model_id] = profile_key
        return self.save()

    def remember_model(self, model_id: str) -> Path:
        self.last_used_model = model_id
        return self.save()

    def save(self) -> Path:
        target = Path(self.path).expanduser()
        self.last_updated = datetime.now().isoformat(timespec="seconds")
        payload = {
            "version": PREFERENCES_VERSION,
            "last_used_model": self.last_used_model,
            "preferred_profiles": dict(sorted(self.preferred_profiles.items())),
            "last_updated": self.last_updated,
        }
        temp = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, target)
        except OSError as exc:
            raise ConfigError(
                f"偏好保存失败：{target}；{exc.strerror or exc}"
            ) from exc
        finally:
            if temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass
        return target


def default_preferences_path() -> Path:
    return Path.home() / ".vanitas" / "preferences.json"
