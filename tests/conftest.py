import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import Settings, load_settings  # noqa: E402
from app.storage.repositories import SessionRepository  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    s = load_settings()
    s.session.default_duration_seconds = 1000
    s.recording.output_dir = "data/sessions"
    return s


@pytest.fixture()
def engine_factory(settings, tmp_path):
    """Create engines that write sessions into pytest's tmp dir."""
    from app.main import BabyMotionEngine

    created = []

    def factory(enable_perception: bool = False) -> BabyMotionEngine:
        repo = SessionRepository(json_dir=tmp_path)
        e = BabyMotionEngine(settings, enable_perception=enable_perception,
                             repository=repo)
        created.append(e)
        return e

    yield factory
    for e in created:
        e.close()
